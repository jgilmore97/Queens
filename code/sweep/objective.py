import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import optuna
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
from train import FocalLoss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import HRM
from data_loader import (
    QueensDataset,
    MixedDataset,
    create_filtered_old_dataset
)
from sweep.vectorized_eval import evaluate_solve_rate


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def create_model_from_trial(trial: optuna.Trial) -> HRM:
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 160, 192])
    gat_heads = trial.suggest_categorical('gat_heads', [2, 4])
    hgt_heads = trial.suggest_categorical('hgt_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    z_dim = trial.suggest_categorical('z_dim', [128, 256])

    model = HRM(
        input_dim=14,
        hidden_dim=hidden_dim,
        gat_heads=gat_heads,
        hgt_heads=hgt_heads,
        dropout=dropout,
        use_batch_norm=True,
        n_cycles=3,
        t_micro=2,
        use_input_injection=True,
        z_dim=z_dim,
    )

    return model

def train_epoch_simple(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Simplified training epoch for sweep (no W&B, minimal logging)."""
    model.train()
    total_loss = 0.0
    total_nodes = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
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

        num_nodes = len(labels)
        total_loss += loss.item() * num_nodes
        total_nodes += num_nodes

        pbar.set_postfix({'loss': f'{total_loss/total_nodes:.4f}'})

    return {'loss': total_loss / total_nodes}


def objective(
    trial: optuna.Trial,
    train_json: str = "data/StateTrainingSet.json",
    state0_json: str = "data/State0TrainingSet.json",
    val_json: str = "data/StateValSet.json",
    device: str = "cuda",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> float:
    """
    Optuna objective function.
    Trains HRM for 6 epochs (2 pre-switch + 4 post-switch).
    """
    # Resolve paths relative to project root
    root = get_project_root()
    train_json = str(root / train_json)
    state0_json = str(root / state0_json)
    val_json = str(root / val_json)

    # Sample training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    focal_alpha = trial.suggest_float('focal_alpha', 0.15, 0.5)
    focal_gamma = trial.suggest_float('focal_gamma', 1.5, 3.0)
    mixed_ratio = trial.suggest_float('mixed_ratio', 0.5, 0.7)

    try:
        model = create_model_from_trial(trial)
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nTrial {trial.number}: {param_count:,} parameters")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Pre-switch training 
        train_dataset = QueensDataset(
            train_json,
            split="train",
            val_ratio=val_ratio,
            seed=seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        for epoch in range(1, 3):
            metrics = train_epoch_simple(model, train_loader, criterion, optimizer, device, epoch)
            print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}")

        eval_results = evaluate_solve_rate(
            model,
            val_json_path=val_json,
            device=device,
            batch_size=128,
            val_ratio=val_ratio,
            seed=seed
        )

        epoch2_solve_rate = eval_results['solve_rate']
        print(f"  Epoch 2 solve_rate: {epoch2_solve_rate:.4f}")

        trial.report(epoch2_solve_rate, step=2)
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch 2")
            raise optuna.TrialPruned()

        # Post-switch training
        # Halve learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2

        state0_dataset = QueensDataset(
            state0_json,
            split="train",
            val_ratio=val_ratio,
            seed=seed
        )
        filtered_old_dataset = create_filtered_old_dataset(
            train_json,
            val_ratio=val_ratio,
            seed=seed,
            split="train"
        )
        mixed_dataset = MixedDataset(
            state0_dataset,
            filtered_old_dataset,
            ratio1=mixed_ratio
        )
        mixed_loader = DataLoader(
            mixed_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        for epoch in range(3, 7):
            metrics = train_epoch_simple(model, mixed_loader, criterion, optimizer, device, epoch)
            print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}")

        eval_results = evaluate_solve_rate(
            model,
            val_json_path=val_json,
            device=device,
            batch_size=128,
            val_ratio=val_ratio,
            seed=seed
        )
        epoch6_solve_rate = eval_results['solve_rate']
        print(f"  Epoch 6 solve_rate: {epoch6_solve_rate:.4f}")

        trial.set_user_attr('epoch2_solve_rate', epoch2_solve_rate)
        trial.set_user_attr('epoch6_solve_rate', epoch6_solve_rate)
        trial.set_user_attr('status', 'completed')

        return epoch6_solve_rate

    except optuna.TrialPruned:
        trial.set_user_attr('epoch2_solve_rate', epoch2_solve_rate)
        trial.set_user_attr('epoch6_solve_rate', None)
        trial.set_user_attr('status', 'pruned')
        raise

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  Trial {trial.number} OOM - marking as failed")
            torch.cuda.empty_cache()
            trial.set_user_attr('epoch2_solve_rate', None)
            trial.set_user_attr('epoch6_solve_rate', None)
            trial.set_user_attr('status', 'oom')
            return 0.0
        else:
            raise

    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        traceback.print_exc()
        trial.set_user_attr('epoch2_solve_rate', None)
        trial.set_user_attr('epoch6_solve_rate', None)
        trial.set_user_attr('status', f'error: {str(e)[:50]}')
        return 0.0

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()