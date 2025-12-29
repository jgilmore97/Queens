import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm.auto import tqdm
import numpy as np

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
from model import GAT, HeteroGAT, HRM
from sweep.vectorized_eval import evaluate_solve_rate


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

def train_epoch_hetero(model, loader, criterion, optimizer, device, epoch):
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

        # if hasattr(batch, 'x_dict'):
        logits = model(batch)
        labels = batch['cell'].y
        num_nodes = batch['cell'].num_nodes
        # else:
        #     x_dict = {'cell': batch['cell'].x}
        #     edge_index_dict = {
        #         ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
        #         ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
        #         ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
        #     }
        #     logits = model(x_dict, edge_index_dict)
        #     labels = batch['cell'].y
        #     num_nodes = len(labels)

        loss = criterion(logits, labels.float())

        loss.backward()
        optimizer.step()

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
def evaluate_epoch_hetero(model, loader, criterion, device, epoch):
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

        if hasattr(batch, 'x_dict'):
            logits = model(batch)
            labels = batch['cell'].y
            num_nodes = batch['cell'].num_nodes
        else:
            x_dict = {'cell': batch['cell'].x}
            edge_index_dict = {
                ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
                ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
                ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
            }
            logits = model(x_dict, edge_index_dict)
            labels = batch['cell'].y
            num_nodes = len(labels)

        loss = criterion(logits, labels.float())

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
    """Create learning rate scheduler based on config."""
    if config.training.scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer,
                               T_max=config.training.cosine_t_max,
                               eta_min=config.training.cosine_eta_min)
    elif config.training.scheduler_type == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.training.scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif config.training.scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {config.training.scheduler_type}")

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
        if config.training.switch_epoch < config.training.epochs:
            print(f"Will switch to mixed dataset (75% state-0, 25% old) at epoch {config.training.switch_epoch}")

        for epoch in range(1, config.training.epochs + 1):
            if epoch == config.training.switch_epoch and mixed_train_loader is None:
                print(f"\n🔄 Switching to mixed dataset at epoch {epoch}")
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

                # Halve learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2
                print(f"Divided learning rate by 2. Now: {optimizer.param_groups[0]['lr']:.1e}")

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
        print(f"📊 Key insight: Top-1 accuracy shows how well argmax(logits) performs")

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


        print(f"Starting HETEROGENEOUS training for {config.training.epochs} epochs")
        print(f"Device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Scheduler: {config.training.scheduler_type}")
        if config.training.scheduler_type == "cosine":
            print(f"  T_max: {config.training.cosine_t_max}, eta_min: {config.training.cosine_eta_min:.1e}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")

        for epoch in range(1, config.training.epochs + 1):
            current_lr = optimizer.param_groups[0]['lr']
            tracker.set_current_lr(current_lr)

            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if epoch_start_time:
                epoch_start_time.record()

            train_metrics = train_epoch_hetero(model, train_loader, criterion, optimizer, device, epoch)
            val_metrics = evaluate_epoch_hetero(model, val_loader, criterion, device, epoch)

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

            F1_CONVERGENCE_THRESHOLD = 0.9935
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
