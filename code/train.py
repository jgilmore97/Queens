import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from collections import Counter
from tqdm.auto import tqdm
import numpy as np

# Import our FIXED tracking - note the new import name
from experiment_tracker_fixed import ExperimentTracker
from config import Config

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch - clean and efficient."""
    model.train()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0
    
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

        # Metrics calculation
        preds = (logits > 0).long()
        batch_correct = (preds == batch.y).sum().item()
        batch_TP = ((preds == 1) & (batch.y == 1)).sum().item()
        batch_FP = ((preds == 1) & (batch.y == 0)).sum().item()
        batch_FN = ((preds == 0) & (batch.y == 1)).sum().item()
        batch_TN = ((preds == 0) & (batch.y == 0)).sum().item()
        
        # Accumulate
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN
        
        batch_losses.append(loss.item())
        
        # Update progress bar (visual feedback only)
        current_loss = total_loss / total_nodes
        current_acc = correct / total_nodes
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.3f}'
        })
        
    
    # Calculate epoch metrics
    eps = 1e-9
    avg_loss = total_loss / total_nodes
    accuracy = correct / total_nodes
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss_std': np.std(batch_losses)
    }
    
    return metrics

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, epoch):
    """Evaluate model for one epoch."""
    model.eval()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)
    
    for batch in pbar:
        batch = batch.to(device)
        
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y.float())
        
        # Predictions and probabilities
        probs = torch.sigmoid(logits)
        preds = (logits > 0).long()
        
        # Store predictions for detailed analysis
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        
        # Metrics
        batch_correct = (preds == batch.y).sum().item()
        batch_TP = ((preds == 1) & (batch.y == 1)).sum().item()
        batch_FP = ((preds == 1) & (batch.y == 0)).sum().item()
        batch_FN = ((preds == 0) & (batch.y == 1)).sum().item()
        batch_TN = ((preds == 0) & (batch.y == 0)).sum().item()
        
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN
        
        # Update progress bar (this is just visual feedback, not W&B logging)
        current_loss = total_loss / total_nodes
        current_acc = correct / total_nodes
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.3f}'
        })
    
    # Calculate metrics
    eps = 1e-9
    avg_loss = total_loss / total_nodes
    accuracy = correct / total_nodes
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # Additional analysis
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Confidence statistics
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
        'label_positive_rate': all_labels.mean()
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
    """Main training loop with FIXED consolidated experiment tracking."""
    
    # Initialize FIXED experiment tracker
    tracker = ExperimentTracker(config, resume_id=resume_id)
    
    try:
        device = config.system.device
        model = model.to(device)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        scheduler = create_scheduler(optimizer, config)
        
        # Setup loss function
        criterion = FocalLoss(
            alpha=config.training.focal_alpha, 
            gamma=config.training.focal_gamma
        )
        
        # Training state
        best_val_f1 = 0.0
        best_epoch = 0
        
        print(f"Starting training for {config.training.epochs} epochs")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(1, config.training.epochs + 1):
            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if epoch_start_time:
                epoch_start_time.record()
            
            # Training (removed tracker parameter)
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            
            # Validation
            val_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch)
            
            # Scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['f1'])
            else:
                scheduler.step()
            
            # Update learning rate in tracker
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.training.learning_rate
            tracker.set_current_lr(current_lr)
            
            if epoch_end_time:
                epoch_end_time.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0  # seconds
                train_metrics['epoch_time_seconds'] = epoch_time
            
            # CONSOLIDATED LOGGING - Single call with all metrics
            tracker.log_epoch_metrics(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
                model=model,  # For gradient logging
                val_loader=val_loader,  # For prediction logging
                device=device
            )
            
            # Check for best model
            is_best = val_metrics['f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
            
            # Save checkpoint (local only)
            tracker.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
            
            # Print progress
            print(f"Epoch {epoch:02d} | "
                  f"Train: L={train_metrics['loss']:.4f} F1={train_metrics['f1']:.3f} | "
                  f"Val: L={val_metrics['loss']:.4f} F1={val_metrics['f1']:.3f} Acc={val_metrics['accuracy']*100:.1f}% | "
                  f"LR={current_lr:.1e} | {'🎯' if is_best else ''}")
        
        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
        
        return model, best_val_f1
    
    finally:
        # Always clean up tracker
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