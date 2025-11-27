"""
HRM Hyperparameter Tuning with Optuna

Optimizes HRM model hyperparameters using Bayesian optimization,
with full puzzle solve rate as the primary objective.

Usage (Google Colab):
    !python code/hyperparameter_tuning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

from config import HRM_TUNING_SPACE, HRM_TUNING_CONFIG
from model import HRM
from data_loader import (
    QueensDataset,
    MixedDataset,
    create_filtered_old_dataset,
    build_heterogeneous_edge_index,
)

# Disable W&B for hyperparameter tuning (use Optuna's own logging)
os.environ['WANDB_MODE'] = 'disabled'

# Configure Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

# Global cached data loader (created once, reused across trials)
_cached_train_loader = None

# Global cached validation puzzles for full-solve evaluation
_cached_full_solve_puzzles = None


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
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slower)")
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


def create_train_loader(tuning_config: dict, force_recreate: bool = False, use_full_data: bool = False):
    """Create training data loader for tuning (state0-heavy mixed dataset).

    Uses caching to avoid recreating loader for each trial.

    Args:
        tuning_config: Configuration dictionary
        force_recreate: Force recreation even if cached
        use_full_data: If True, use full dataset (for final retraining)
    """
    global _cached_train_loader

    # Return cached loader if available
    if not force_recreate and _cached_train_loader is not None:
        return _cached_train_loader

    print("Creating training data loader (will be cached for subsequent trials)...")

    batch_size = tuning_config["batch_size"]
    mixed_ratio = tuning_config["mixed_ratio"]
    state0_path = tuning_config["state0_json_path"]
    multistate_path = tuning_config["multistate_json_path"]
    subsample_ratio = tuning_config.get("train_subsample_ratio", 1.0)

    # Create state-0 dataset (use all data, no val split needed)
    state0_train = QueensDataset(
        state0_path,
        split="train",
        val_ratio=0.0,  # Use all data for training
        seed=SEED
    )

    # Create filtered multi-state dataset
    multistate_train = create_filtered_old_dataset(
        multistate_path,
        val_ratio=0.0,  # Use all data for training
        seed=SEED,
        split="train"
    )

    # Create mixed dataset (state0-heavy)
    mixed_train = MixedDataset(state0_train, multistate_train, mixed_ratio)

    # Apply subsampling for faster tuning trials (unless using full data for retraining)
    if not use_full_data and subsample_ratio and subsample_ratio < 1.0:
        full_size = len(mixed_train)
        subset_size = int(full_size * subsample_ratio)
        rng = np.random.RandomState(SEED)
        indices = rng.choice(full_size, size=subset_size, replace=False)
        mixed_train = Subset(mixed_train, indices)
        print(f"  Subsampling training data: {subset_size:,}/{full_size:,} ({subsample_ratio:.0%})")

    # Use num_workers=0 for Colab compatibility
    train_loader = DataLoader(
        mixed_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        follow_batch=[]
    )

    # Cache the loader
    _cached_train_loader = train_loader

    print(f"  Train samples: {len(train_loader.dataset):,}")

    return train_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return loss."""
    model.train()
    total_loss = 0.0
    total_nodes = 0

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

    return total_loss / total_nodes


def _build_node_features(region_board: np.ndarray, queen_board: np.ndarray, max_regions: int = 11) -> torch.Tensor:
    """Build node feature vectors for evaluation (same as Solver._build_node_features)."""
    n = region_board.shape[0]
    N2 = n * n

    coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)

    reg_onehot = np.zeros((N2, max_regions), dtype=np.float32)
    flat_ids = region_board.flatten()
    reg_onehot[np.arange(N2), flat_ids] = 1.0

    has_queen = queen_board.flatten()[:, None].astype(np.float32)

    features = np.hstack([coords, reg_onehot, has_queen])
    return torch.from_numpy(features)


def _get_full_solve_puzzles(full_solve_path: str) -> list:
    """Load and cache full-solve validation puzzles."""
    global _cached_full_solve_puzzles

    if _cached_full_solve_puzzles is not None:
        return _cached_full_solve_puzzles

    try:
        with open(full_solve_path, 'r') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return []

    # Filter to state-0 puzzles only
    state_0_puzzles = []
    for puzzle in test_data:
        if 'step' in puzzle and puzzle['step'] == 0:
            state_0_puzzles.append(puzzle)
        elif 'step' not in puzzle:
            state_0_puzzles.append(puzzle)

    _cached_full_solve_puzzles = state_0_puzzles
    return state_0_puzzles


@torch.no_grad()
def evaluate_full_solve(model, params: dict, full_solve_path: str, device: torch.device,
                        max_puzzles: int = None) -> float:
    """Evaluate model on full puzzle solving and return success rate.

    This version evaluates directly without creating a Solver object,
    avoiding disk I/O and redundant model loading.

    Args:
        model: The HRM model to evaluate
        params: Model hyperparameters (not used in direct evaluation)
        full_solve_path: Path to validation puzzles JSON
        device: Torch device
        max_puzzles: If set, evaluate only this many puzzles (for faster tuning)
    """
    model.eval()

    state_0_puzzles = _get_full_solve_puzzles(full_solve_path)

    if not state_0_puzzles:
        return 0.0

    # Subsample puzzles for faster evaluation during tuning
    if max_puzzles and max_puzzles < len(state_0_puzzles):
        # Use deterministic sampling for consistency across trials
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(state_0_puzzles), size=max_puzzles, replace=False)
        state_0_puzzles = [state_0_puzzles[i] for i in indices]

    successful_solves = 0

    for puzzle in state_0_puzzles:
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]

        if 'label_board' not in puzzle:
            continue
        expected_solution = np.array(puzzle['label_board'])

        # Build edge index once per puzzle
        edge_index_dict = build_heterogeneous_edge_index(region_board)
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

        edge_index_dict_formatted = {
            ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
            ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
            ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
        }

        correct_positions = [(r, c) for r in range(n) for c in range(n) if expected_solution[r, c] == 1]
        queen_board = np.zeros((n, n), dtype=int)

        is_perfect = True

        for step in range(n):
            node_features = _build_node_features(region_board, queen_board)
            node_features = node_features.to(device)

            x_dict = {'cell': node_features}
            logits = model(x_dict, edge_index_dict_formatted)

            logits_np = logits.cpu().numpy().reshape(n, n)
            top_idx = np.argmax(logits_np.flatten())
            top_row, top_col = top_idx // n, top_idx % n

            # Check if placement is correct
            remaining_correct = [pos for pos in correct_positions if queen_board[pos[0], pos[1]] == 0]
            is_correct = (top_row, top_col) in remaining_correct

            if not is_correct:
                is_perfect = False
                break

            queen_board[top_row, top_col] = 1

        if is_perfect:
            successful_solves += 1

    success_rate = successful_solves / len(state_0_puzzles) if state_0_puzzles else 0.0
    return success_rate


def cleanup_gpu():
    """Clean up GPU memory between trials."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function: maximize full puzzle solve rate."""
    device = get_device()
    tuning_config = HRM_TUNING_CONFIG

    params = sample_hyperparameters(trial)

    # Log trial parameters
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing configuration")
    print(f"{'='*60}")
    for k, v in params.items():
        if k not in HRM_TUNING_SPACE.get("fixed", {}):
            print(f"  {k}: {v}")

    model = None
    try:
        model = create_model(params, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
    except Exception as e:
        print(f"  Model creation failed: {e}")
        cleanup_gpu()
        raise optuna.TrialPruned()

    try:
        train_loader = create_train_loader(tuning_config)
    except Exception as e:
        print(f"  Data loading failed: {e}")
        cleanup_gpu()
        raise optuna.TrialPruned()

    try:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        epochs = tuning_config["epochs_per_trial"]
        eval_every = tuning_config["eval_every_n_epochs"]
        pruning_warmup = tuning_config["pruning_warmup_epochs"]
        full_solve_path = tuning_config["full_solve_val_path"]
        eval_subsample = tuning_config.get("eval_subsample_size")

        best_solve_rate = 0.0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step(train_loss)

            if epoch % eval_every == 0 or epoch == epochs:
                solve_rate = evaluate_full_solve(model, params, full_solve_path, device,
                                                 max_puzzles=eval_subsample)
                best_solve_rate = max(best_solve_rate, solve_rate)

                print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Solve: {solve_rate:.1%}")

                trial.report(solve_rate, epoch)

                if epoch >= pruning_warmup and trial.should_prune():
                    print(f"  Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
            else:
                print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f}")

        print(f"  Trial {trial.number} completed | Best solve rate: {best_solve_rate:.1%}")
        return best_solve_rate

    except torch.cuda.OutOfMemoryError:
        print(f"  Trial {trial.number} OOM - skipping this configuration")
        raise optuna.TrialPruned()

    finally:
        # Always clean up GPU memory after trial
        del model
        cleanup_gpu()


def retrain_best_config(best_params: dict, device: torch.device, save_dir: Path):
    """Retrain the best configuration with full epochs."""
    tuning_config = HRM_TUNING_CONFIG
    full_epochs = tuning_config["full_training_epochs"]

    print(f"\n{'='*70}")
    print("RETRAINING BEST CONFIGURATION")
    print(f"{'='*70}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    model = create_model(best_params, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Force recreate loader for final training (fresh shuffle, FULL data - no subsampling)
    train_loader = create_train_loader(tuning_config, force_recreate=True, use_full_data=True)
    print(f"\nTraining for {full_epochs} epochs (using full dataset)...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    full_solve_path = tuning_config["full_solve_val_path"]

    best_solve_rate = 0.0
    best_model_state = None

    for epoch in range(1, full_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)

        if epoch % 2 == 0 or epoch == full_epochs:
            solve_rate = evaluate_full_solve(model, best_params, full_solve_path, device)

            is_best = solve_rate > best_solve_rate
            if is_best:
                best_solve_rate = solve_rate
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"Epoch {epoch:02d}/{full_epochs} | Loss: {train_loss:.4f} | "
                  f"Solve: {solve_rate:.1%} {'[BEST]' if is_best else ''}")
        else:
            print(f"Epoch {epoch:02d}/{full_epochs} | Loss: {train_loss:.4f}")

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

    print("\n" + "="*70)
    print("HRM HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per trial: {tuning_config['epochs_per_trial']}")
    print(f"Full training epochs: {tuning_config['full_training_epochs']}")
    print(f"Results directory: {results_dir}")
    print(f"Primary metric: Full puzzle solve rate")
    print("="*70)

    # Print search space
    print("\nSearch Space:")
    for name, config in HRM_TUNING_SPACE.items():
        if name == "fixed":
            continue
        if config["type"] == "categorical":
            print(f"  {name}: {config['values']}")
        else:
            print(f"  {name}: [{config['low']}, {config['high']}] ({config['type']})")

    print("\nFixed Parameters:")
    for name, value in HRM_TUNING_SPACE.get("fixed", {}).items():
        print(f"  {name}: {value}")

    # Print subsampling settings
    train_subsample = tuning_config.get("train_subsample_ratio", 1.0)
    eval_subsample = tuning_config.get("eval_subsample_size")
    if train_subsample and train_subsample < 1.0:
        print(f"\nSubsampling (faster tuning):")
        print(f"  Training data: {train_subsample:.0%} of full dataset")
    if eval_subsample:
        print(f"  Eval puzzles: {eval_subsample} puzzles per evaluation")
    print("="*70 + "\n")

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Create Optuna study
    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=tuning_config["pruning_warmup_epochs"],
        interval_steps=1
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'='*70}")
    print("TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
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
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
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

    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best configuration solve rate: {final_solve_rate:.1%}")
    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {results_dir / 'best_tuned_hrm.pt'}")
    print(f"\nNext step: Use the tuned model for inference or further training")
    print("="*70)

    return study, final_model


if __name__ == "__main__":
    run_hyperparameter_tuning()
