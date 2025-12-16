"""Training scripts for benchmark models."""

import random
import numpy as np
import torch
from config import Config
from data_loader import get_benchmark_loaders
from bm_model import BenchmarkComparisonModel
from train import FocalLoss, create_scheduler
from experiment_tracker import ExperimentTracker
from tqdm.auto import tqdm


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

def calculate_top1_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1
) -> tuple[int, int]:
    """Calculate top-k accuracy, only considering valid (non-padded) positions."""
    batch_size = logits.size(0)

    # Mask out padded positions by setting their logits to -inf
    valid_mask = labels >= 0  # [batch, seq_len]
    masked_logits = logits.clone()
    masked_logits[~valid_mask] = float('-inf')

    _, topk_indices = torch.topk(masked_logits, k=k, dim=-1)  # [batch, k]
    topk_labels = torch.gather(labels, dim=1, index=topk_indices)  # [batch, k]
    topk_correct = (topk_labels == 1).any(dim=1).sum().item()

    return topk_correct, batch_size

def train_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """Train the model for one epoch. Non-graph based."""
    
    model.train()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0

    total_top1_correct = 0
    total_top1_predictions = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.squeeze(-1)

        # Create valid mask (exclude padded positions with y=-1)
        valid_mask = y >= 0
        valid_logits = logits[valid_mask]
        valid_y = y[valid_mask]

        # Compute loss only on valid positions
        loss = criterion(valid_logits, valid_y.float())
        loss.backward()
        optimizer.step()

        # Compute metrics only on valid positions
        preds = (valid_logits > 0).long()
        batch_correct = (preds == valid_y).sum().item()
        batch_TP = ((preds == 1) & (valid_y == 1)).sum().item()
        batch_FP = ((preds == 1) & (valid_y == 0)).sum().item()
        batch_FN = ((preds == 0) & (valid_y == 1)).sum().item()
        batch_TN = ((preds == 0) & (valid_y == 0)).sum().item()
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN
        num_valid = valid_mask.sum().item()
        total_loss += loss.item() * num_valid
        total_nodes += num_valid

        # Top-1 metrics
        top1_correct, top1_predictions,  = calculate_top1_metrics(logits, y)
        total_top1_correct += top1_correct
        total_top1_predictions += top1_predictions

        pbar.set_postfix({
            'Loss': f"{total_loss / total_nodes:.4f}",
            'Acc': f"{correct / total_nodes:.4f}",
            'Precision': f"{TP / (TP + FP + 1e-8):.4f}",
            'Recall': f"{TP / (TP + FN + 1e-8):.4f}"
        })

    eps = 1e-8
    avg_loss = total_loss / total_nodes
    accuracy = correct / total_nodes
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    top1_accuracy = total_top1_correct / total_top1_predictions

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top1_accuracy': top1_accuracy
    }

    return metrics

@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """Evaluate the model for one epoch. Non-graph based."""
    
    model.eval()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0

    total_top1_correct = 0
    total_top1_predictions = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Evaluation", leave=False)
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        logits = model(x)
        logits = logits.squeeze(-1)

        # Create valid mask (exclude padded positions with y=-1)
        valid_mask = y >= 0
        valid_logits = logits[valid_mask]
        valid_y = y[valid_mask]

        # Compute loss only on valid positions
        loss = criterion(valid_logits, valid_y.float())

        # Compute metrics only on valid positions
        preds = (valid_logits > 0).long()
        batch_correct = (preds == valid_y).sum().item()
        batch_TP = ((preds == 1) & (valid_y == 1)).sum().item()
        batch_FP = ((preds == 1) & (valid_y == 0)).sum().item()
        batch_FN = ((preds == 0) & (valid_y == 1)).sum().item()
        batch_TN = ((preds == 0) & (valid_y == 0)).sum().item()
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN
        num_valid = valid_mask.sum().item()
        total_loss += loss.item() * num_valid
        total_nodes += num_valid

        # Top-1 metrics
        top1_correct, top1_predictions = calculate_top1_metrics(logits, y)
        total_top1_correct += top1_correct
        total_top1_predictions += top1_predictions

        pbar.set_postfix({
            'Loss': f"{total_loss / total_nodes:.4f}",
            'Acc': f"{correct / total_nodes:.4f}",
            'Precision': f"{TP / (TP + FP + 1e-8):.4f}",
            'Recall': f"{TP / (TP + FN + 1e-8):.4f}"
        })  

    eps = 1e-8
    avg_loss = total_loss / total_nodes
    accuracy = correct / total_nodes
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    top1_accuracy = total_top1_correct / total_top1_predictions

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top1_accuracy': top1_accuracy,
    }

    return metrics

def benchmark_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Config
):
    # Set seeds for reproducibility
    set_seed(config.data.seed)

    tracker = ExperimentTracker(config)

    try:
        device = config.system.device
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay)
        
        criterion = FocalLoss(
            alpha=config.training.focal_alpha, 
            gamma=config.training.focal_gamma
            )  
        
        scheduler = create_scheduler(
            optimizer, 
            config
        )

        best_f1 = 0.0
        best_top1_acc = 0.0
        for epoch in range(1, config.training.epochs + 1):
            train_metrics = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                device, 
                epoch
            )
            val_metrics = evaluate_epoch(
                model, 
                val_loader, 
                criterion, 
                device, 
                epoch
            )

            tracker.log_epoch_metrics(train_metrics, val_metrics, epoch)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                tracker.save_checkpoint(model, optimizer, epoch, val_metrics, is_best=True)
            if val_metrics['top1_accuracy'] > best_top1_acc:
                best_top1_acc = val_metrics['top1_accuracy']
                best_top1_epoch = epoch

            scheduler.step(val_metrics['f1'])

            base_log = (f"Epoch {epoch:02d}| "
                       f"Train: L={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.3f} F1={train_metrics['f1']:.3f} T1={train_metrics['top1_accuracy']:.3f} | "
                       f"Val: L={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.3f} F1={val_metrics['f1']:.3f} T1={val_metrics['top1_accuracy']:.3f}")
            print(base_log)

        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Best validation Top-1 Accuracy: {best_top1_acc:.4f} (epoch {best_top1_epoch})")

        return model, best_f1
    
    finally:
        tracker.finish()
        
        
