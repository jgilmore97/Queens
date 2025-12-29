import random
import numpy as np
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import optuna
from typing import Dict, Any
from pathlib import Path
import traceback

from model import HRM
from data_loader import (
    QueensDataset,
    get_combined_queens_loaders,
    SizeBucketBatchSampler,
)
from train import FocalLoss
from sweep.vectorized_eval import evaluate_solve_rate


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def create_model_from_trial(trial: optuna.Trial) -> HRM:
    """Create HRM with hyperparameters sampled from trial."""
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 160])
    gat_heads = trial.suggest_categorical('gat_heads', [2, 4])
    hgt_heads = trial.suggest_categorical('hgt_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.08, 0.3)
    z_dim = trial.suggest_categorical('z_dim', [64, 128, 256])
    n_cycles = 2
    t_micro = trial.suggest_categorical('t_micro', [2, 3])

    model = HRM(
        input_dim=14,
        hidden_dim=hidden_dim,
        gat_heads=gat_heads,
        hgt_heads=hgt_heads,
        dropout=dropout,
        use_batch_norm=False,
        n_cycles=n_cycles,
        t_micro=t_micro,
        use_input_injection=True,
        z_dim=z_dim,
        use_hmod=False,
        same_size_batches=False,
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
    """Simplified training epoch for sweep."""
    model.train()
    total_loss = 0.0
    total_nodes = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        labels = batch['cell'].y

        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()

        num_nodes = batch['cell'].num_nodes
        total_loss += loss.item() * num_nodes
        total_nodes += num_nodes

        pbar.set_postfix({'loss': f'{total_loss/total_nodes:.4f}'})

    return {'loss': total_loss / total_nodes}


def objective(
    trial: optuna.Trial,
    train_json: str = "/data/StateTrainingSet.json",
    state0_json: str = "/data/State0TrainingSet.json",
    val_json: str = "/data/StateValSet.json",
    device: str = "cuda",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> float:
    """
    Optuna objective function.
    Trains HRM using combined dataset for specified epochs.
    Returns solve rate on validation set.
    """
    set_seed(seed)

    root = get_project_root()
    train_json = str(root / train_json)
    state0_json = str(root / state0_json)
    val_json = str(root / val_json)

    # Sample training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    focal_alpha = trial.suggest_float('focal_alpha', 0.15, 0.40)
    focal_gamma = trial.suggest_float('focal_gamma', 1.5, 3.0)
    batch_size = 512

    num_epochs = 4

    epoch2_solve_rate = None

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

        train_loader, val_loader = get_combined_queens_loaders(
            train_json,
            state0_json,
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=4,
            pin_memory=True,
            shuffle_train=True,
            same_size_batches=False,
            drop_last=False,
        )

        for epoch in range(1, num_epochs + 1):
            metrics = train_epoch_simple(model, train_loader, criterion, optimizer, device, epoch)
            print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}")

            # Early checkpoint for pruning
            # if epoch == 4:
            #     eval_results = evaluate_solve_rate(
            #         model,
            #         val_json_path=val_json,
            #         device=device,
            #         batch_size=128,
            #         val_ratio=val_ratio,
            #         seed=seed,
            #     )
            #     epoch2_solve_rate = eval_results['solve_rate']
            #     print(f"  Epoch 4 solve_rate: {epoch2_solve_rate:.4f}")

            #     trial.report(epoch2_solve_rate, step=4)
            #     if trial.should_prune():
            #         print(f"  Trial {trial.number} pruned at epoch 4")
            #         raise optuna.TrialPruned()

        eval_results = evaluate_solve_rate(
            model,
            val_json_path=val_json,
            device=device,
            batch_size=128,
            val_ratio=val_ratio,
            seed=seed,
        )
        final_solve_rate = eval_results['solve_rate']
        print(f"  Final solve_rate: {final_solve_rate:.4f}")

        # trial.set_user_attr('epoch4_solve_rate', epoch2_solve_rate)
        trial.set_user_attr('final_solve_rate', final_solve_rate)
        trial.set_user_attr('param_count', param_count)
        trial.set_user_attr('status', 'completed')

        return final_solve_rate

    except optuna.TrialPruned:
        trial.set_user_attr('epoch4_solve_rate', epoch2_solve_rate)
        trial.set_user_attr('final_solve_rate', None)
        trial.set_user_attr('status', 'pruned')
        raise

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  Trial {trial.number} OOM")
            torch.cuda.empty_cache()
            trial.set_user_attr('status', 'oom')
            return 0.0
        raise

    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        traceback.print_exc()
        trial.set_user_attr('status', f'error: {str(e)[:50]}')
        return 0.0

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()