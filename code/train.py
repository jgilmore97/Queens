import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, SequentialLR, LinearLR, ConstantLR
from tqdm.auto import tqdm
import numpy as np
from torch.cuda.amp import autocast

from experiment_tracker import ExperimentTracker
from data_loader import (
    QueensDataset,
    HomogeneousQueensDataset,
    MixedDataset,
    create_filtered_old_dataset,
    create_filtered_old_dataset_homogeneous,
    get_combined_queens_loaders,
)
from config import Config
from model import GAT, HeteroGAT, HRM, HRM_FullSpatial
from vectorized_eval import evaluate_solve_rate


def calculate_top1_metrics(logits, labels, batch_info):
    """Calculate top-1 accuracy, recall, and precision metrics using vectorized operations."""
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
    true_positives = correct_top1

    valid_mask = logits_matrix > float('-inf')
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

def calculate_top1_metrics_hetero(logits, labels, batch_info):
    """Calculate top-1 metrics for heterogeneous graph data."""
    device = logits.device

    batch_indices = batch_info['cell'].batch

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
    true_positives = correct_top1

    valid_mask = logits_matrix > float('-inf')
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
    """Train for one epoch with threshold-based and top-1 metrics."""
    model.train()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0

    total_top1_correct = 0
    total_top1_tp = 0
    total_top1_predictions = 0
    total_top1_positive_labels = 0

    batch_losses = []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y.float())

        loss.backward()
        optimizer.step()

        preds = (logits > 0).long()
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

def train_epoch_hetero(model, loader, criterion, optimizer, device, epoch, use_amp=True):
    """Train for one epoch with top-1 metrics for heterogeneous data."""
    model.train()
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0

    total_top1_correct = 0
    total_top1_tp = 0
    total_top1_predictions = 0
    total_top1_positive_labels = 0

    batch_losses = []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(batch)

            labels = batch['cell'].y
            num_nodes = batch['cell'].num_nodes
            loss = criterion(logits, labels.float())

        loss.backward()
        optimizer.step()

        logits = logits.float()

        preds = (logits > 0).long()
        batch_correct = (preds == labels).sum().item()
        batch_TP = ((preds == 1) & (labels == 1)).sum().item()
        batch_FP = ((preds == 1) & (labels == 0)).sum().item()
        batch_FN = ((preds == 0) & (labels == 1)).sum().item()
        batch_TN = ((preds == 0) & (labels == 0)).sum().item()

        top1_metrics = calculate_top1_metrics_hetero(logits, labels, batch)

        total_loss += loss.item() * num_nodes
        total_nodes += num_nodes
        correct += batch_correct
        TP += batch_TP
        FP += batch_FP
        FN += batch_FN
        TN += batch_TN

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
    """Evaluate model for one epoch with threshold-based and top-1 metrics."""
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

@torch.no_grad()
def evaluate_epoch_hetero(model, loader, criterion, device, epoch, use_amp=True):
    """Evaluate model for one epoch with top-1 metrics for heterogeneous data."""
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

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(batch)

            labels = batch['cell'].y
            num_nodes = batch['cell'].num_nodes
            loss = criterion(logits, labels.float())

        logits = logits.float()

        probs = torch.sigmoid(logits)
        preds = (logits > 0).long()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        batch_correct = (preds == labels).sum().item()
        batch_TP = ((preds == 1) & (labels == 1)).sum().item()
        batch_FP = ((preds == 1) & (labels == 0)).sum().item()
        batch_FN = ((preds == 0) & (labels == 1)).sum().item()
        batch_TN = ((preds == 0) & (labels == 0)).sum().item()

        top1_metrics = calculate_top1_metrics_hetero(logits, labels, batch)

        total_loss += loss.item() * num_nodes
        total_nodes += num_nodes
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
    """Create learning rate scheduler with optional warmup and constant tail."""
    warmup_epochs = getattr(config.training, 'warmup_epochs', 0)
    constant_epochs = getattr(config.training, 'constant_lr_epochs', 0)
    
    if config.training.scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif config.training.scheduler_type == "none":
        return None
    elif config.training.scheduler_type == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.training.scheduler_type != "cosine":
        raise ValueError(f"Unknown scheduler type: {config.training.scheduler_type}")
    
    schedulers = []
    milestones = []
    
    cosine_epochs = config.training.cosine_t_max - warmup_epochs
    
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=config.training.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_epochs)
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cosine_epochs),
        eta_min=config.training.cosine_eta_min
    )
    schedulers.append(cosine_scheduler)
    
    if constant_epochs > 0:
        milestones.append(warmup_epochs + cosine_epochs)
        constant_lr_value = getattr(config.training, 'constant_lr', None) or config.training.cosine_eta_min
        constant_factor = constant_lr_value / config.training.learning_rate
        constant_scheduler = ConstantLR(
            optimizer,
            factor=constant_factor,
            total_iters=constant_epochs
        )
        schedulers.append(constant_scheduler)
    
    if len(schedulers) == 1:
        return schedulers[0]
    
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

def run_training_with_tracking(model, train_loader, val_loader, config, resume_id=None):
    """Main training loop with experiment tracking and top-1 metrics."""

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

        mixed_train_loader, state0_val_loader = None, None
        current_dataset = "multi-state"
        switched = False

        print(f"Starting training for {config.training.epochs} epochs")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Tracking both threshold-based and top-1 metrics")
        print("Using HOMOGENEOUS graph format (all edge types merged)")

        switch_epoch = getattr(config.training, 'switch_epoch', None)
        if switch_epoch and switch_epoch < config.training.epochs:
            print(f"Will switch to mixed dataset (75% state-0, 25% old) at epoch {switch_epoch}")

        for epoch in range(1, config.training.epochs + 1):
            if switch_epoch and epoch == switch_epoch and mixed_train_loader is None:
                print(f"\n Switching to mixed dataset at epoch {epoch}")
                print(f"Loading state-0 dataset from {config.training.state0_json_path}")

                # Create homogeneous datasets
                state0_train_dataset = HomogeneousQueensDataset(
                    config.training.state0_json_path,
                    split="train",
                    val_ratio=config.training.val_ratio,
                    seed=config.data.seed
                )
                state0_val_dataset = HomogeneousQueensDataset(
                    config.training.state0_json_path,
                    split="val",
                    val_ratio=config.training.val_ratio,
                    seed=config.data.seed
                )

                filtered_old_train_dataset = create_filtered_old_dataset_homogeneous(
                    config.data.train_json,
                    val_ratio=config.training.val_ratio,
                    seed=config.data.seed,
                    split="train"
                )

                mixed_dataset = MixedDataset(
                    state0_train_dataset,
                    filtered_old_train_dataset,
                    config.training.mixed_ratio
                )

                mixed_train_loader = DataLoader(
                    mixed_dataset,
                    batch_size=config.training.batch_size // 2,
                    num_workers=config.data.num_workers,
                    pin_memory=config.data.pin_memory,
                    shuffle=True,
                )

                state0_val_loader = DataLoader(
                    state0_val_dataset,
                    batch_size=config.training.batch_size // 4,
                    num_workers=config.data.num_workers,
                    pin_memory=config.data.pin_memory,
                    shuffle=False,
                )

                current_dataset = "mixed (75% state-0, 25% old)"
                switched = True

                print(f"Mixed train samples: {len(mixed_dataset):,}")
                print(f"State-0 val samples: {len(state0_val_dataset):,}")

            if switched and mixed_train_loader is not None:
                active_train_loader = mixed_train_loader
                current_dataset = "mixed (75% state-0, 25% old)"
            else:
                active_train_loader = train_loader
                current_dataset = "multi-state"

            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if epoch_start_time:
                epoch_start_time.record()

            train_metrics = train_epoch(model, active_train_loader, criterion, optimizer, device, epoch)

            val_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch)
            if state0_val_loader is not None:
                state0_val_metrics = evaluate_epoch(model, state0_val_loader, criterion, device, epoch)
            else:
                state0_val_metrics = None

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['f1'])
            else:
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.training.learning_rate
            tracker.set_current_lr(current_lr)

            if epoch_end_time:
                epoch_end_time.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
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

            base_log = (f"Epoch {epoch:02d} [{current_dataset}] | "
                       f"Train: L={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.3f} F1={train_metrics['f1']:.3f} T1={train_metrics['top1_accuracy']:.3f} | "
                       f"Val: L={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.3f} F1={val_metrics['f1']:.3f} T1={val_metrics['top1_accuracy']:.3f}")

            if state0_val_metrics is not None:
                base_log += f" | State0-Val: Acc={state0_val_metrics['accuracy']:.3f} F1={state0_val_metrics['f1']:.3f} T1={state0_val_metrics['top1_accuracy']:.3f}"

            base_log += f" | LR={current_lr:.1e} | {'🎯F1' if is_best_f1 else ''}{'🎯T1' if is_best_top1 else ''}"
            print(base_log)

        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
        print(f"Best validation Top-1 Accuracy: {best_val_top1:.4f} (epoch {best_top1_epoch})")
        print(f"Key insight: Top-1 accuracy shows how well argmax(logits) performs")

        return model, best_val_f1

    finally:
        tracker.finish()

def run_training_with_tracking_hetero(model, train_loader, val_loader, config, resume_id=None):
    """Main training loop for heterogeneous models."""

    tracker = ExperimentTracker(config, resume_id=resume_id)

    try:
        device = config.system.device
        model = model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        criterion = FocalLoss(
            alpha=config.training.focal_alpha,
            gamma=config.training.focal_gamma
        )

        scheduler = create_scheduler(optimizer, config)

        best_val_f1 = 0.0
        best_val_top1 = 0.0
        best_solve_rate = 0.0
        best_epoch = 0
        best_top1_epoch = 0
        is_best_solve_rate = False
        best_solve_rate = 0.0
        is_best = False

        use_amp = config.system.mixed_precision and torch.cuda.is_available()

        print(f"Starting HETEROGENEOUS training for {config.training.epochs} epochs")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Scheduler: {config.training.scheduler_type}")
        if config.training.scheduler_type == "cosine":
            print(f"T_max: {config.training.cosine_t_max}, eta_min: {config.training.cosine_eta_min:.1e}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        for epoch in range(1, config.training.epochs + 1):
            current_lr = optimizer.param_groups[0]['lr']
            tracker.set_current_lr(current_lr)

            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if epoch_start_time:
                epoch_start_time.record()

            train_metrics = train_epoch_hetero(model, train_loader, criterion, optimizer, device, epoch, use_amp=use_amp)
            val_metrics = evaluate_epoch_hetero(model, val_loader, criterion, device, epoch, use_amp=use_amp)

            if epoch_end_time:
                epoch_end_time.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
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

            solve_stats = evaluate_solve_rate(model, config.data.auto_reg_json, device)
            solve_rate = solve_stats['solve_rate']

            if config.model.model_type == "HRM_FullSpatial":
                F1_CONVERGENCE_THRESHOLD = 0.9935
            else:
                F1_CONVERGENCE_THRESHOLD = 0.0
            converged = val_metrics['f1'] >= F1_CONVERGENCE_THRESHOLD

            if converged:
                is_best_solve_rate = solve_rate > best_solve_rate
                is_best = is_best_solve_rate
                if is_best:
                    best_solve_rate = solve_rate
            # else:
            #     is_best = is_best_f1

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(solve_rate)
                else:
                    scheduler.step()

            tracker.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)

            base_log = (f"Epoch {epoch:02d} | "
                       f"Train: L={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.3f} "
                       f"F1={train_metrics['f1']:.3f} T1={train_metrics['top1_accuracy']:.3f} | "
                       f"Val: L={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.3f} "
                       f"F1={val_metrics['f1']:.3f} T1={val_metrics['top1_accuracy']:.3f} | "
                       f"Solve: {solve_rate:.3f} | LR={current_lr:.1e}")
            
            flags = []
            if is_best_f1:
                flags.append("F1")
            if is_best_top1:
                flags.append("T1")
            if is_best_solve_rate:
                flags.append("Solve")
            if flags:
                base_log += f" | [{'+'.join(flags)}]"

            print(base_log)

        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
        print(f"Best validation Top-1: {best_val_top1:.4f} (epoch {best_top1_epoch})")
        print(f"Best solve rate: {best_solve_rate:.3f}")

        return model, best_val_f1

    finally:
        tracker.finish()


def train_model_for_ablation(
    model_type: str,
    multistate_json: str,
    state0_json: str,
    test_json: str,
    checkpoint_dir: str,
    config_overrides: dict = None,
    device: str = 'cuda'
):
    """
    Generic training orchestrator for ablation studies.

    Args:
        model_type: 'gat', 'hetero_gat', 'hrm_fullspatial', 'benchmark_hrm', or 'benchmark_sequential'
        multistate_json: Path to multi-state training data
        state0_json: Path to state-0 training data
        test_json: Path to test data
        checkpoint_dir: Directory to save checkpoint
        config_overrides: Dict of config overrides
        device: Device to train on

    Returns:
        (model, best_f1, training_time)
    """
    import time
    import os
    from pathlib import Path
    from torch.utils.data import ConcatDataset
    from torch.utils.data import DataLoader as VanillaDataLoader
    from torch_geometric.loader import DataLoader as GraphDataLoader
    from data_loader import HomogeneousQueensDataset, BenchmarkDataset
    from bm_model import BenchmarkHRM, BenchmarkSequential
    from bm_train import benchmark_training

    config = Config()

    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.benchmark, key):
                setattr(config.benchmark, key, value)

    config.data.train_json = multistate_json
    config.data.auto_reg_json = test_json
    config.training.state0_json_path = state0_json
    config.training.combine_state0 = True
    config.experiment.experiment_name = f"ablation_{model_type}"
    config.experiment.checkpoint_dir = checkpoint_dir
    config.experiment.tags = ['ablation', model_type]

    os.environ['WANDB_MODE'] = 'disabled'

    print("\n" + "="*80)
    print(f"TRAINING {model_type.upper()}")
    print("="*80)

    if model_type == 'gat':
        config.model.model_type = 'GAT'
        config.model.layer_count = config_overrides.get('layer_count', 6)

        model = GAT(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            layer_count=config.model.layer_count,
            dropout=config.model.dropout,
            heads=config.model.gat_heads
        )

        ds_multistate_train = HomogeneousQueensDataset(
            multistate_json, split="train", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_multistate_val = HomogeneousQueensDataset(
            multistate_json, split="val", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_state0_train = HomogeneousQueensDataset(
            state0_json, split="train", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_state0_val = HomogeneousQueensDataset(
            state0_json, split="val", val_ratio=config.training.val_ratio, seed=config.data.seed
        )

        combined_train = ConcatDataset([ds_multistate_train, ds_state0_train])
        combined_val = ConcatDataset([ds_multistate_val, ds_state0_val])

        train_loader = GraphDataLoader(combined_train, batch_size=config.training.batch_size,
                                       shuffle=True, num_workers=0, pin_memory=True)
        val_loader = GraphDataLoader(combined_val, batch_size=config.training.batch_size,
                                     shuffle=False, num_workers=0, pin_memory=True)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        start_time = time.time()
        model, best_f1 = run_training_with_tracking(model, train_loader, val_loader, config)
        training_time = time.time() - start_time

    elif model_type == 'hetero_gat':
        config.model.model_type = 'HeteroGAT'
        config.model.layer_count = config_overrides.get('layer_count', 6)

        model = HeteroGAT(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            layer_count=config.model.layer_count,
            dropout=config.model.dropout,
            gat_heads=config.model.gat_heads,
            hgt_heads=config.model.hgt_heads,
            use_batch_norm=True
        )

        train_loader, val_loader = get_combined_queens_loaders(
            multistate_json, state0_json,
            batch_size=config.training.batch_size,
            val_ratio=config.training.val_ratio,
            seed=config.data.seed,
            num_workers=0, pin_memory=True, shuffle_train=True,
            same_size_batches=False, drop_last=False
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        start_time = time.time()
        model, best_f1 = run_training_with_tracking_hetero(model, train_loader, val_loader, config)
        training_time = time.time() - start_time

    elif model_type == 'hrm_fullspatial':
        config.model.model_type = 'HRM_FullSpatial'
        config.model.n_cycles = config_overrides.get('n_cycles', 3)
        config.model.t_micro = config_overrides.get('t_micro', 2)
        config.training.same_size_batches = True
        config.training.drop_last = True

        model = HRM_FullSpatial(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            gat_heads=config.model.gat_heads,
            hgt_heads=config.model.hgt_heads,
            hmod_heads=config.model.hmod_heads,
            dropout=config.model.dropout,
            n_cycles=config.model.n_cycles,
            t_micro=config.model.t_micro,
        )

        train_loader, val_loader = get_combined_queens_loaders(
            multistate_json, state0_json,
            batch_size=config.training.batch_size,
            val_ratio=config.training.val_ratio,
            seed=config.data.seed,
            num_workers=0, pin_memory=True, shuffle_train=True,
            same_size_batches=True, drop_last=True
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"HRM Config: {config.model.n_cycles} cycles, {config.model.t_micro} micro-steps")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        start_time = time.time()
        model, best_f1 = run_training_with_tracking_hetero(model, train_loader, val_loader, config)
        training_time = time.time() - start_time

    elif model_type == 'benchmark_hrm':
        config.benchmark.model_type = 'hrm'
        config.benchmark.n_cycles = config_overrides.get('n_cycles', 3)
        config.benchmark.microsteps = config_overrides.get('t_micro', 2)

        model = BenchmarkHRM(
            input_dim=config.benchmark.input_dim,
            hidden_dim=config.benchmark.hidden_dim,
            p_drop=config.benchmark.dropout,
            n_heads=config.benchmark.n_heads,
            n_cycles=config.benchmark.n_cycles,
            t_micro=config.benchmark.microsteps,
        )

        ds_multistate_train = BenchmarkDataset(
            multistate_json, split="train", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_multistate_val = BenchmarkDataset(
            multistate_json, split="val", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_state0_train = BenchmarkDataset(
            state0_json, split="train", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_state0_val = BenchmarkDataset(
            state0_json, split="val", val_ratio=config.training.val_ratio, seed=config.data.seed
        )

        combined_train = ConcatDataset([ds_multistate_train, ds_state0_train])
        combined_val = ConcatDataset([ds_multistate_val, ds_state0_val])

        train_loader = VanillaDataLoader(combined_train, batch_size=config.training.batch_size,
                                        shuffle=True, num_workers=0, pin_memory=True)
        val_loader = VanillaDataLoader(combined_val, batch_size=config.training.batch_size,
                                      shuffle=False, num_workers=0, pin_memory=True)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        start_time = time.time()
        model, best_f1 = benchmark_training(model, train_loader, val_loader, config)
        training_time = time.time() - start_time

    elif model_type == 'benchmark_sequential':
        config.benchmark.model_type = 'sequential'
        config.benchmark.layers = config_overrides.get('layers', 6)

        model = BenchmarkSequential(
            input_dim=config.benchmark.input_dim,
            hidden_dim=config.benchmark.hidden_dim,
            p_drop=config.benchmark.dropout,
            n_heads=config.benchmark.n_heads,
            layers=config.benchmark.layers,
        )

        ds_multistate_train = BenchmarkDataset(
            multistate_json, split="train", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_multistate_val = BenchmarkDataset(
            multistate_json, split="val", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_state0_train = BenchmarkDataset(
            state0_json, split="train", val_ratio=config.training.val_ratio, seed=config.data.seed
        )
        ds_state0_val = BenchmarkDataset(
            state0_json, split="val", val_ratio=config.training.val_ratio, seed=config.data.seed
        )

        combined_train = ConcatDataset([ds_multistate_train, ds_state0_train])
        combined_val = ConcatDataset([ds_multistate_val, ds_state0_val])

        train_loader = VanillaDataLoader(combined_train, batch_size=config.training.batch_size,
                                        shuffle=True, num_workers=0, pin_memory=True)
        val_loader = VanillaDataLoader(combined_val, batch_size=config.training.batch_size,
                                      shuffle=False, num_workers=0, pin_memory=True)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        start_time = time.time()
        model, best_f1 = benchmark_training(model, train_loader, val_loader, config)
        training_time = time.time() - start_time

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / f'{model_type}_ablation.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_f1': best_f1,
        'config': config.to_dict(),
    }, checkpoint_path)

    print(f"{model_type.upper()} Training Complete! Best F1: {best_f1:.4f} | Time: {training_time/60:.1f} min")
    print(f"Checkpoint saved to {checkpoint_path}")

    return model, best_f1, training_time


class FocalLoss(nn.Module):
    """Binary focal loss for logits."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Compute focal loss from raw logits and float targets."""
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
