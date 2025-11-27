"""
HRM Hyperparameter Tuning with Optuna

Optimizes HRM model hyperparameters using Bayesian optimization,
with full puzzle solve rate as the primary objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

from config import HRM_TUNING_SPACE, HRM_TUNING_CONFIG
from model import HRM
from data_loader import (
    QueensDataset,
    MixedDataset,
    create_filtered_old_dataset,
)
from improved_solver import Solver
from evaluation_util import evaluate_full_puzzle_capability


class FocalLoss(nn.Module):
    """Binary focal loss for logits."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_term = (1 - pt) ** self.gamma
        return (self.alpha * focal_term * bce).mean()


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    """Sample hyperparameters from the search space."""
    params = {}

    for name, config in HRM_TUNING_SPACE.items():
        if name == "fixed":
            continue

        if config["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, config["values"])
        elif config["type"] == "loguniform":
            params[name] = trial.suggest_float(name, config["low"], config["high"], log=True)
        elif config["type"] == "uniform":
            params[name] = trial.suggest_float(name, config["low"], config["high"])
        elif config["type"] == "int":
            params[name] = trial.suggest_int(name, config["low"], config["high"])

    # Add fixed parameters
    params.update(HRM_TUNING_SPACE["fixed"])

    # Validate: hidden_dim must be divisible by gat_heads and h_pooling_heads
    if params["hidden_dim"] % params["gat_heads"] != 0:
        # Adjust gat_heads to be compatible
        for heads in [2, 4]:
            if params["hidden_dim"] % heads == 0:
                params["gat_heads"] = heads
                break

    if params["hidden_dim"] % params["h_pooling_heads"] != 0:
        for heads in [2, 4, 8]:
            if params["hidden_dim"] % heads == 0:
                params["h_pooling_heads"] = heads
                break

    return params


def create_model(params: dict, device: torch.device) -> HRM:
    """Create an HRM model from hyperparameters."""
    model = HRM(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        gat_heads=params["gat_heads"],
        hgt_heads=params["hgt_heads"],
        dropout=params["dropout"],
        use_batch_norm=True,
        n_cycles=params["n_cycles"],
        t_micro=params["t_micro"],
        use_input_injection=params["use_input_injection"],
        z_init=params["z_init"],
        h_pooling_heads=params["h_pooling_heads"],
    )
    return model.to(device)


def create_data_loaders(tuning_config: dict, seed: int = 42):
    """Create data loaders for tuning (state0-heavy mixed dataset)."""
    batch_size = tuning_config["batch_size"]
    mixed_ratio = tuning_config["mixed_ratio"]
    state0_path = tuning_config["state0_json_path"]

    # Create state-0 dataset
    state0_train = QueensDataset(
        state0_path,
        split="train",
        val_ratio=0.10,
        seed=seed
    )
    state0_val = QueensDataset(
        state0_path,
        split="val",
        val_ratio=0.10,
        seed=seed
    )

    # Create filtered multi-state dataset
    multistate_train = create_filtered_old_dataset(
        "10k_training_set_with_states.json",
        val_ratio=0.10,
        seed=seed,
        split="train"
    )

    # Create mixed dataset (state0-heavy)
    mixed_train = MixedDataset(state0_train, multistate_train, mixed_ratio)

    train_loader = DataLoader(
        mixed_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        follow_batch=[]
    )

    val_loader = DataLoader(
        state0_val,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        follow_batch=[]
    )

    return train_loader, val_loader


def calculate_top1_accuracy(logits, labels, batch_info):
    """Calculate top-1 accuracy for a batch of heterogeneous graphs."""
    device = logits.device

    if hasattr(batch_info, 'batch_dict') and 'cell' in batch_info.batch_dict:
        batch_indices = batch_info.batch_dict['cell']
    elif hasattr(batch_info, '_slice_dict') and 'cell' in batch_info._slice_dict:
        slices = batch_info._slice_dict['cell']['x']
        batch_indices = torch.zeros(len(logits), dtype=torch.long, device=device)
        for i in range(len(slices) - 1):
            batch_indices[slices[i]:slices[i+1]] = i
    else:
        batch_indices = torch.zeros(len(logits), dtype=torch.long, device=device)

    unique_batches = torch.unique(batch_indices)
    num_graphs = len(unique_batches)

    correct = 0
    for i in unique_batches:
        mask = (batch_indices == i)
        graph_logits = logits[mask]
        graph_labels = labels[mask]
        top_idx = torch.argmax(graph_logits)
        if graph_labels[top_idx].item() == 1:
            correct += 1

    return correct / num_graphs if num_graphs > 0 else 0.0


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return metrics."""
    model.train()
    total_loss = 0.0
    total_nodes = 0
    total_top1_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        x_dict = {'cell': batch['cell'].x}
        edge_index_dict = {
            ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
            ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
            ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
        }
        labels = batch['cell'].y

        logits = model(x_dict, edge_index_dict)
        loss = criterion(logits, labels.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_nodes += len(labels)

        # Calculate top-1 accuracy
        with torch.no_grad():
            top1_acc = calculate_top1_accuracy(logits, labels, batch)
            # Estimate graph count from batch
            if hasattr(batch, 'batch_dict') and 'cell' in batch.batch_dict:
                n_graphs = len(torch.unique(batch.batch_dict['cell']))
            else:
                n_graphs = 1
            total_top1_correct += top1_acc * n_graphs
            total_graphs += n_graphs

    return {
        'loss': total_loss / total_nodes,
        'top1_accuracy': total_top1_correct / total_graphs if total_graphs > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    total_top1_correct = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        x_dict = {'cell': batch['cell'].x}
        edge_index_dict = {
            ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
            ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
            ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
        }
        labels = batch['cell'].y

        logits = model(x_dict, edge_index_dict)
        loss = criterion(logits, labels.float())

        total_loss += loss.item() * len(labels)
        total_nodes += len(labels)

        top1_acc = calculate_top1_accuracy(logits, labels, batch)
        if hasattr(batch, 'batch_dict') and 'cell' in batch.batch_dict:
            n_graphs = len(torch.unique(batch.batch_dict['cell']))
        else:
            n_graphs = 1
        total_top1_correct += top1_acc * n_graphs
        total_graphs += n_graphs

    return {
        'loss': total_loss / total_nodes,
        'top1_accuracy': total_top1_correct / total_graphs if total_graphs > 0 else 0.0,
    }


def evaluate_full_solve(model, params: dict, full_solve_path: str, device: torch.device) -> float:
    """Evaluate model on full puzzle solving and return success rate."""
    # Save model temporarily for Solver to load
    temp_checkpoint_path = Path(HRM_TUNING_CONFIG["results_dir"]) / "temp_model.pt"
    temp_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create config dict matching what Solver expects
    config_dict = {
        'input_dim': params['input_dim'],
        'hidden_dim': params['hidden_dim'],
        'gat_heads': params['gat_heads'],
        'hgt_heads': params['hgt_heads'],
        'dropout': params['dropout'],
        'n_cycles': params['n_cycles'],
        't_micro': params['t_micro'],
        'use_input_injection': params['use_input_injection'],
        'z_init': params['z_init'],
        'model_type': 'HRM',
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'config_dict': config_dict,
    }, temp_checkpoint_path)

    try:
        solver = Solver(str(temp_checkpoint_path), device=str(device))
        stats = evaluate_full_puzzle_capability(solver, full_solve_path, verbose=False)
        success_rate = stats.get('success_rate', 0.0)
    except Exception as e:
        print(f"Full-solve evaluation failed: {e}")
        success_rate = 0.0
    finally:
        # Clean up temp file
        if temp_checkpoint_path.exists():
            temp_checkpoint_path.unlink()

    return success_rate


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function: maximize full puzzle solve rate."""
    device = get_device()
    tuning_config = HRM_TUNING_CONFIG

    # Sample hyperparameters
    params = sample_hyperparameters(trial)

    # Log trial parameters
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing configuration")
    print(f"{'='*60}")
    for k, v in params.items():
        if k not in HRM_TUNING_SPACE.get("fixed", {}):
            print(f"  {k}: {v}")

    # Create model
    try:
        model = create_model(params, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
    except Exception as e:
        print(f"Model creation failed: {e}")
        raise optuna.TrialPruned()

    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(tuning_config)
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise optuna.TrialPruned()

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    epochs = tuning_config["epochs_per_trial"]
    eval_every = tuning_config["eval_every_n_epochs"]
    pruning_warmup = tuning_config["pruning_warmup_epochs"]
    full_solve_path = tuning_config["full_solve_val_path"]

    best_solve_rate = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (single-step)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics['top1_accuracy'])

        # Full-solve evaluation (expensive, do less frequently)
        if epoch % eval_every == 0 or epoch == epochs:
            solve_rate = evaluate_full_solve(model, params, full_solve_path, device)
            best_solve_rate = max(best_solve_rate, solve_rate)

            print(f"Epoch {epoch:02d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} T1: {train_metrics['top1_accuracy']:.3f} | "
                  f"Val T1: {val_metrics['top1_accuracy']:.3f} | "
                  f"Full-Solve: {solve_rate:.1%}")

            # Report to Optuna for pruning
            trial.report(solve_rate, epoch)

            # Check if should prune
            if epoch >= pruning_warmup and trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        else:
            print(f"Epoch {epoch:02d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} T1: {train_metrics['top1_accuracy']:.3f} | "
                  f"Val T1: {val_metrics['top1_accuracy']:.3f}")

    print(f"Trial {trial.number} completed with best solve rate: {best_solve_rate:.1%}")
    return best_solve_rate


def retrain_best_config(best_params: dict, device: torch.device, save_dir: Path):
    """Retrain the best configuration with full epochs."""
    tuning_config = HRM_TUNING_CONFIG
    full_epochs = tuning_config["full_training_epochs"]

    print(f"\n{'='*60}")
    print("RETRAINING BEST CONFIGURATION")
    print(f"{'='*60}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Create model
    model = create_model(best_params, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(tuning_config)

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    full_solve_path = tuning_config["full_solve_val_path"]

    best_solve_rate = 0.0
    best_model_state = None

    for epoch in range(1, full_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics['top1_accuracy'])

        # Evaluate full-solve every 2 epochs
        if epoch % 2 == 0 or epoch == full_epochs:
            solve_rate = evaluate_full_solve(model, best_params, full_solve_path, device)

            print(f"Epoch {epoch:02d}/{full_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} T1: {train_metrics['top1_accuracy']:.3f} | "
                  f"Val T1: {val_metrics['top1_accuracy']:.3f} | "
                  f"Full-Solve: {solve_rate:.1%}")

            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                best_model_state = model.state_dict().copy()
                print(f"  New best solve rate!")
        else:
            print(f"Epoch {epoch:02d}/{full_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} T1: {train_metrics['top1_accuracy']:.3f} | "
                  f"Val T1: {val_metrics['top1_accuracy']:.3f}")

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    checkpoint_path = save_dir / "best_tuned_hrm.pt"
    config_dict = {
        'input_dim': best_params['input_dim'],
        'hidden_dim': best_params['hidden_dim'],
        'gat_heads': best_params['gat_heads'],
        'hgt_heads': best_params['hgt_heads'],
        'dropout': best_params['dropout'],
        'n_cycles': best_params['n_cycles'],
        't_micro': best_params['t_micro'],
        'use_input_injection': best_params['use_input_injection'],
        'z_init': best_params['z_init'],
        'h_pooling_heads': best_params['h_pooling_heads'],
        'model_type': 'HRM',
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'config_dict': config_dict,
        'best_solve_rate': best_solve_rate,
        'hyperparameters': best_params,
    }, checkpoint_path)

    print(f"\nBest model saved to {checkpoint_path}")
    print(f"Best full-solve rate: {best_solve_rate:.1%}")

    return model, best_solve_rate


def run_hyperparameter_tuning():
    """Main entry point for hyperparameter tuning."""
    tuning_config = HRM_TUNING_CONFIG
    n_trials = tuning_config["n_trials"]

    # Create results directory
    results_dir = Path(tuning_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"hrm_tuning_{timestamp}"

    print(f"\n{'='*60}")
    print("HRM HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per trial: {tuning_config['epochs_per_trial']}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}\n")

    # Create Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=tuning_config["pruning_warmup_epochs"],
        interval_steps=1
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize solve rate
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best solve rate: {study.best_value:.1%}")
    print(f"\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save study results
    results = {
        'study_name': study_name,
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'elapsed_time_seconds': elapsed_time,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ]
    }

    results_path = results_dir / f"{study_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Retrain best configuration with full epochs
    device = get_device()
    best_params = {**study.best_params, **HRM_TUNING_SPACE["fixed"]}

    # Ensure compatibility
    if best_params["hidden_dim"] % best_params["gat_heads"] != 0:
        for heads in [2, 4]:
            if best_params["hidden_dim"] % heads == 0:
                best_params["gat_heads"] = heads
                break

    if best_params["hidden_dim"] % best_params["h_pooling_heads"] != 0:
        for heads in [2, 4, 8]:
            if best_params["hidden_dim"] % heads == 0:
                best_params["h_pooling_heads"] = heads
                break

    final_model, final_solve_rate = retrain_best_config(best_params, device, results_dir)

    # Update results with final training
    results['final_solve_rate'] = final_solve_rate
    results['final_params'] = best_params
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best configuration solve rate: {final_solve_rate:.1%}")
    print(f"Model saved to: {results_dir / 'best_tuned_hrm.pt'}")

    return study, final_model


if __name__ == "__main__":
    run_hyperparameter_tuning()
