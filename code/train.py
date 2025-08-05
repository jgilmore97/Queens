import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from collections import Counter
from tqdm.auto import tqdm
import numpy as np

from experiment_tracker_fixed import ExperimentTracker
from config import Config

def calculate_top1_metrics(logits, labels, batch_info):
    """
    Vectorized calculation of top-1 accuracy and recall metrics.
    
    Args:
        logits: Raw model outputs [N] where N is total nodes across batch
        labels: Ground truth labels [N]
        batch_info: Batch object with .batch attribute for grouping nodes by graph
    
    Returns:
        dict with top1_accuracy, top1_recall, top1_precision
    """
    device = logits.device
    
    if hasattr(batch_info, 'batch'):
        batch_indices = batch_info.batch
    else:
        batch_indices = torch.zeros(len(logits), dtype=torch.long, device=device)
    
    unique_batches, inverse_indices = torch.unique(batch_indices, return_inverse=True)
    num_graphs = len(unique_batches)
    

    max_nodes = torch.bincount(batch_indices).max().item()
    
    logits_matrix = torch.full((num_graphs, max_nodes), float('-inf'), device=device)
    labels_matrix = torch.zeros((num_graphs, max_nodes), dtype=torch.long, device=device)
    
    graph_positions = torch.zeros_like(batch_indices)
    for i in range(num_graphs):
        mask = (batch_indices == unique_batches[i])
        graph_positions[mask] = torch.arange(mask.sum(), device=device)
    
    logits_matrix[inverse_indices, graph_positions] = logits
    labels_matrix[inverse_indices, graph_positions] = labels
    
    top_pred_indices = torch.argmax(logits_matrix, dim=1)  # [num_graphs]
    
    top_pred_labels = labels_matrix[torch.arange(num_graphs, device=device), top_pred_indices]
    
    correct_top1 = (top_pred_labels == 1).sum().item()
    true_positives = correct_top1  # Same thing for top-1
    
    valid_mask = logits_matrix > float('-inf')  # Ignore padding
    positive_counts = (labels_matrix * valid_mask).sum(dim=1)  # [num_graphs]
    total_positive_labels = positive_counts.sum().item()
    
    top1_accuracy = correct_top1 / num_graphs if num_graphs > 0 else 0.0
    top1_recall = true_positives / total_positive_labels if total_positive_labels > 0 else 0.0
    top1_precision = true_positives / num_graphs if num_graphs > 0 else 0.0
    
    return {
        'top1_accuracy': top1_accuracy,
        'top1_recall': top1_recall, 
        'top1_precision': top1_precision,
        'total_graphs': num_graphs,
        'avg_positive_per_graph': total_positive_labels / num_graphs if num_graphs > 0 else 0.0
    }

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch with enhanced metrics including top-1."""
    model.train()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0
    
    # Top-1 metrics accumulation
    total_top1_correct = 0
    total_top1_tp = 0
    total_top1_predictions = 0
    total_top1_positive_labels = 0
    
    batch_losses = []
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Standard threshold-based metrics
        preds = (logits > 0).long()
        batch_correct = (preds == batch.y).sum().item()
        batch_TP = ((preds == 1) & (batch.y == 1)).sum().item()
        batch_FP = ((preds == 1) & (batch.y == 0)).sum().item()
        batch_FN = ((preds == 0) & (batch.y == 1)).sum().item()
        batch_TN = ((preds == 0) & (batch.y == 0)).sum().item()
        
        # Top-1 metrics for this batch
        top1_metrics = calculate_top1_metrics(logits, batch.y, batch)
        
        # Accumulate standard metrics
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN
        
        # Accumulate top-1 metrics
        total_top1_correct += top1_metrics['top1_accuracy'] * top1_metrics['total_graphs']
        total_top1_tp += top1_metrics['top1_recall'] * top1_metrics['avg_positive_per_graph'] * top1_metrics['total_graphs']
        total_top1_predictions += top1_metrics['total_graphs']
        total_top1_positive_labels += top1_metrics['avg_positive_per_graph'] * top1_metrics['total_graphs']
        
        batch_losses.append(loss.item())
        
        current_loss = total_loss / total_nodes
        current_acc = correct / total_nodes
        current_top1 = total_top1_correct / total_top1_predictions if total_top1_predictions > 0 else 0
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.3f}',
            'top1': f'{current_top1:.3f}'
        })
        
    
    eps = 1e-9
    avg_loss = total_loss / total_nodes
    accuracy = correct / total_nodes
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    top1_accuracy = total_top1_correct / total_top1_predictions if total_top1_predictions > 0 else 0.0
    top1_recall = total_top1_tp / total_top1_positive_labels if total_top1_positive_labels > 0 else 0.0
    top1_precision = total_top1_tp / total_top1_predictions if total_top1_predictions > 0 else 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss_std': np.std(batch_losses),
        'top1_accuracy': top1_accuracy,
        'top1_recall': top1_recall,
        'top1_precision': top1_precision
    }
    
    return metrics

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, epoch):
    """Evaluate model for one epoch with enhanced metrics including top-1."""
    model.eval()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0
    
    total_top1_correct = 0
    total_top1_tp = 0
    total_top1_predictions = 0
    total_top1_positive_labels = 0
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)
    
    for batch in pbar:
        batch = batch.to(device)
        
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y.float())
        
        probs = torch.sigmoid(logits)
        preds = (logits > 0).long()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        
        batch_correct = (preds == batch.y).sum().item()
        batch_TP = ((preds == 1) & (batch.y == 1)).sum().item()
        batch_FP = ((preds == 1) & (batch.y == 0)).sum().item()
        batch_FN = ((preds == 0) & (batch.y == 1)).sum().item()
        batch_TN = ((preds == 0) & (batch.y == 0)).sum().item()
        
        top1_metrics = calculate_top1_metrics(logits, batch.y, batch)
        
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN
        
        total_top1_correct += top1_metrics['top1_accuracy'] * top1_metrics['total_graphs']
        total_top1_tp += top1_metrics['top1_recall'] * top1_metrics['avg_positive_per_graph'] * top1_metrics['total_graphs']
        total_top1_predictions += top1_metrics['total_graphs']
        total_top1_positive_labels += top1_metrics['avg_positive_per_graph'] * top1_metrics['total_graphs']
        
        current_loss = total_loss / total_nodes
        current_acc = correct / total_nodes
        current_top1 = total_top1_correct / total_top1_predictions if total_top1_predictions > 0 else 0
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.3f}',
            'top1': f'{current_top1:.3f}'
        })
    
    eps = 1e-9
    avg_loss = total_loss / total_nodes
    accuracy = correct / total_nodes
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    top1_accuracy = total_top1_correct / total_top1_predictions if total_top1_predictions > 0 else 0.0
    top1_recall = total_top1_tp / total_top1_positive_labels if total_top1_positive_labels > 0 else 0.0
    top1_precision = total_top1_tp / total_top1_predictions if total_top1_predictions > 0 else 0.0
    top1_f1 = 2 * top1_precision * top1_recall / (top1_precision + top1_recall + eps)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_confidence = all_probs.mean()
    pos_confidence = all_probs[all_labels == 1].mean() if (all_labels == 1).sum() > 0 else 0
    neg_confidence = 1 - all_probs[all_labels == 0].mean() if (all_labels == 0).sum() > 0 else 0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': avg_confidence,
        'pos_confidence': pos_confidence,
        'neg_confidence': neg_confidence,
        'positive_rate': all_preds.mean(),
        'label_positive_rate': all_labels.mean(),
        'top1_accuracy': top1_accuracy,
        'top1_recall': top1_recall,
        'top1_precision': top1_precision,
        'top1_f1': top1_f1
    }
    
    return metrics

def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    if config.training.scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, 
                               T_max=config.training.cosine_t_max, 
                               eta_min=config.training.cosine_eta_min)
    elif config.training.scheduler_type == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.training.scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        raise ValueError(f"Unknown scheduler type: {config.training.scheduler_type}")

def run_training_with_tracking(model, train_loader, val_loader, config, resume_id=None):
    """Main training loop with enhanced experiment tracking including top-1 metrics."""
    
    tracker = ExperimentTracker(config, resume_id=resume_id)
    
    try:
        device = config.system.device
        model = model.to(device)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        scheduler = create_scheduler(optimizer, config)
        
        criterion = FocalLoss(
            alpha=config.training.focal_alpha, 
            gamma=config.training.focal_gamma
        )
        
        best_val_f1 = 0.0
        best_val_top1 = 0.0
        best_epoch = 0
        best_top1_epoch = 0
        
        print(f"Starting training for {config.training.epochs} epochs")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Tracking both threshold-based and top-1 metrics")
        
        for epoch in range(1, config.training.epochs + 1):
            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if epoch_start_time:
                epoch_start_time.record()
            
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            
            val_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch)
            
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['f1'])
            else:
                scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.training.learning_rate
            tracker.set_current_lr(current_lr)
            
            if epoch_end_time:
                epoch_end_time.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0  # seconds
                train_metrics['epoch_time_seconds'] = epoch_time
            
            tracker.log_epoch_metrics(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
                model=model,
                val_loader=val_loader,
                device=device
            )
            
            is_best_f1 = val_metrics['f1'] > best_val_f1
            is_best_top1 = val_metrics['top1_accuracy'] > best_val_top1
            
            if is_best_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
            
            if is_best_top1:
                best_val_top1 = val_metrics['top1_accuracy']
                best_top1_epoch = epoch
            
            tracker.save_checkpoint(model, optimizer, epoch, val_metrics, is_best_f1 or is_best_top1)
            
            print(f"Epoch {epoch:02d} | "
                  f"Train: L={train_metrics['loss']:.4f} F1={train_metrics['f1']:.3f} T1={train_metrics['top1_accuracy']:.3f} | "
                  f"Val: L={val_metrics['loss']:.4f} F1={val_metrics['f1']:.3f} T1={val_metrics['top1_accuracy']:.3f} | "
                  f"LR={current_lr:.1e} | "
                  f"{'🎯F1' if is_best_f1 else ''}{'🎯T1' if is_best_top1 else ''}")
        
        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
        print(f"Best validation Top-1 Accuracy: {best_val_top1:.4f} (epoch {best_top1_epoch})")
        print(f"📊 Key insight: Top-1 accuracy shows how well argmax(logits) performs")
        
        return model, best_val_f1
    
    finally:
        tracker.finish()

class FocalLoss(nn.Module):
    """Binary focal loss for logits."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : shape [N]  (raw scores, BEFORE sigmoid)
        targets : shape [N]  (float 0/1)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_term = (1 - pt) ** self.gamma
        
        loss = self.alpha * focal_term * bce

        if self.reduction == "mean": 
            return loss.mean()
        elif self.reduction == "sum":  
            return loss.sum()
        else:                          
            return loss