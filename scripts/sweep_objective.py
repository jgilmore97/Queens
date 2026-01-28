import random
import numpy as np
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import optuna
from typing import Dict
from pathlib import Path
import traceback

from queens_solver.models.models import HRM
from queens_solver.data.dataset import get_combined_queens_loaders
from queens_solver.training.trainer import FocalLoss
from queens_solver.evaluation.evaluator import evaluate_solve_rate
from queens_solver.config import Config


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


def create_model_from_trial(trial: optuna.Trial, config: Config) -> HRM:
    """Create HRM with hyperparameters sampled from trial."""
    hidden_dim = 128
    gat_heads = trial.suggest_categorical('gat_heads', [2, 4])
    hgt_heads = trial.suggest_categorical('hgt_heads', [4, 8])
    hmod_heads = trial.suggest_categorical('hmod_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.08, 0.3)

    model = HRM(
        input_dim=config.model.input_dim,
        hidden_dim=hidden_dim,
        gat_heads=gat_heads,
        hgt_heads=hgt_heads,
        hmod_heads=hmod_heads,
        dropout=dropout,
        n_cycles=config.model.n_cycles,
        t_micro=config.model.t_micro
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
    Optuna objective function for HRM.
    Uses Config for fixed parameters, trial for hyperparameters being searched.
    Returns solve rate on validation set.
    """
    config = Config()
    
    set_seed(seed)

    root = get_project_root()
    train_json = str(root / train_json)
    state0_json = str(root / state0_json)
    val_json = str(root / val_json)

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    focal_alpha = trial.suggest_float('focal_alpha', 0.15, 0.40)
    focal_gamma = trial.suggest_float('focal_gamma', 1.5, 3.0)

    num_epochs = 4

    try:
        model = create_model_from_trial(trial, config)
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nTrial {trial.number}: {param_count:,} parameters")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        train_loader, _ = get_combined_queens_loaders(
            train_json,
            state0_json,
            batch_size=config.training.batch_size,
            val_ratio=config.training.val_ratio,
            seed=seed,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            shuffle_train=config.data.shuffle_train,
            same_size_batches=config.training.same_size_batches,
            drop_last=config.training.drop_last,
        )

        for epoch in range(1, num_epochs + 1):
            metrics = train_epoch_simple(model, train_loader, criterion, optimizer, device, epoch)
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")

        eval_results = evaluate_solve_rate(
            model,
            val_json_path=val_json,
            device=device,
            batch_size=config.training.batch_size // 4,
            val_ratio=config.training.val_ratio,
            seed=seed,
        )
        final_solve_rate = eval_results['solve_rate']
        print(f"Final solve_rate: {final_solve_rate:.4f}")

        trial.set_user_attr('final_solve_rate', final_solve_rate)
        trial.set_user_attr('param_count', param_count)
        trial.set_user_attr('status', 'completed')

        return final_solve_rate

    except optuna.TrialPruned:
        trial.set_user_attr('final_solve_rate', None)
        trial.set_user_attr('status', 'pruned')
        raise

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number} OOM")
            torch.cuda.empty_cache()
            trial.set_user_attr('status', 'oom')
            return 0.0
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        trial.set_user_attr('status', f'error: {str(e)[:50]}')
        return 0.0

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()