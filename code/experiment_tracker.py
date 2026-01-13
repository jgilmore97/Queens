"""Experiment tracking with W&B using consolidated epoch-level logging only."""

import os
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import wandb

class ExperimentTracker:
    """Experiment tracking with consolidated epoch-level W&B logging to avoid step conflicts."""

    def __init__(self, config, resume_id: Optional[str] = None):
        self.config = config
        self.start_time = time.time()

        self._init_wandb(resume_id)

        self.process = psutil.Process()
        self.peak_memory = 0

        self._current_lr = config.training.learning_rate

        Path(config.experiment.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.experiment.log_dir).mkdir(exist_ok=True)

        print("Experiment tracker initialized with consolidated logging")

    def _init_wandb(self, resume_id: Optional[str]):
        """Initialize W&B with config and optional resume."""
        simple_config = {
            # Model architecture
            "model_type": self.config.model.model_type,
            "input_dim": self.config.model.input_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "layer_count": self.config.model.layer_count,
            "dropout": self.config.model.dropout,
            "gat_heads": self.config.model.gat_heads,
            "hgt_heads": self.config.model.hgt_heads,
            "hmod_heads": getattr(self.config.model, 'hmod_heads', None),
            "n_cycles": getattr(self.config.model, 'n_cycles', None),
            "t_micro": getattr(self.config.model, 't_micro', None),
            
            # Training
            "epochs": self.config.training.epochs,
            "batch_size": self.config.training.batch_size,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "focal_alpha": self.config.training.focal_alpha,
            "focal_gamma": self.config.training.focal_gamma,
            
            # Scheduler
            "scheduler_type": self.config.training.scheduler_type,
            "cosine_t_max": self.config.training.cosine_t_max,
            "cosine_eta_min": self.config.training.cosine_eta_min,
            "warmup_epochs": getattr(self.config.training, 'warmup_epochs', 0),
            "warmup_start_factor": getattr(self.config.training, 'warmup_start_factor', 0.1),
            
            # System
            "device": self.config.system.device,
            "mixed_precision": getattr(self.config.system, 'mixed_precision', False),
        }
        
        # Remove None values for cleaner logging
        simple_config = {k: v for k, v in simple_config.items() if v is not None}

        init_args = {
            "project": self.config.experiment.project_name,
            "name": self.config.experiment.experiment_name,
            "tags": self.config.experiment.tags or [],
            "notes": self.config.experiment.notes,
            "config": simple_config,
        }

        if hasattr(self.config.experiment, 'entity') and self.config.experiment.entity is not None:
            init_args["entity"] = self.config.experiment.entity

        if resume_id is not None:
            init_args["resume"] = "allow"
            init_args["id"] = resume_id

        wandb.init(**init_args)

        self._log_system_info()

    def _log_system_info(self):
        """Log system information at step 0."""
        system_metrics = {
            "system/device": str(self.config.system.device),
            "system/cuda_available": torch.cuda.is_available(),
            "system/cpu_count": psutil.cpu_count(),
            "system/memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

        if torch.cuda.is_available():
            try:
                system_metrics.update({
                    "system/gpu_name": torch.cuda.get_device_name(),
                    "system/gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                })
            except:
                pass

        wandb.log(system_metrics, step=0)

    def log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int,
                         model: Optional[torch.nn.Module] = None, val_loader=None, device: str = None):
        """Log all epoch-level metrics using epoch number as W&B step."""
        all_metrics = {}

        for k, v in train_metrics.items():
            all_metrics[f"train/{k}"] = v
        for k, v in val_metrics.items():
            all_metrics[f"val/{k}"] = v

        all_metrics["epoch"] = epoch
        all_metrics["learning_rate"] = self._current_lr

        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, memory_mb)
        all_metrics["system/memory_mb"] = memory_mb
        all_metrics["system/peak_memory_mb"] = self.peak_memory

        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                all_metrics["system/gpu_memory_mb"] = gpu_memory_mb
            except:
                pass

        if (model is not None and
            epoch % self.config.experiment.log_gradients_every_n_epochs == 0):
            grad_metrics = self._compute_gradient_metrics(model)
            all_metrics.update(grad_metrics)

        if (model is not None and val_loader is not None and device is not None and
            epoch % self.config.experiment.log_predictions_every_n_epochs == 0):
            pred_metrics = self._compute_prediction_metrics(model, val_loader, device)
            all_metrics.update(pred_metrics)

        wandb.log(all_metrics, step=epoch)

    def _compute_gradient_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute layer-wise gradient statistics for vanishing gradient analysis."""
        grad_stats = {}

        layer_grads = {}
        layer_names = []

        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()

                layer_name = self._extract_layer_name(name)

                if layer_name not in layer_grads:
                    layer_grads[layer_name] = {
                        'grad_norm': 0.0,
                        'param_norm': 0.0,
                        'param_count': 0
                    }

                layer_grads[layer_name]['grad_norm'] += grad_norm ** 2
                layer_grads[layer_name]['param_norm'] += param_norm ** 2
                layer_grads[layer_name]['param_count'] += param.numel()

                layer_names.append(layer_name)

        layer_grad_norms = []
        layer_update_ratios = []

        for layer_name, stats in layer_grads.items():
            layer_grad_norm = (stats['grad_norm'] ** 0.5)
            layer_param_norm = (stats['param_norm'] ** 0.5)

            # Update ratio: how much this layer will change relative to current values
            update_ratio = layer_grad_norm / (layer_param_norm + 1e-8)

            grad_stats[f"gradients/layer_{layer_name}_norm"] = layer_grad_norm
            grad_stats[f"gradients/layer_{layer_name}_update_ratio"] = update_ratio
            grad_stats[f"gradients/layer_{layer_name}_param_count"] = stats['param_count']

            layer_grad_norms.append(layer_grad_norm)
            layer_update_ratios.append(update_ratio)

        if len(layer_grad_norms) >= 2:
            first_layer_grad = layer_grad_norms[0]
            last_layer_grad = layer_grad_norms[-1]

            grad_stats["gradients/first_last_ratio"] = first_layer_grad / (last_layer_grad + 1e-8)
            grad_stats["gradients/first_layer_norm"] = first_layer_grad
            grad_stats["gradients/last_layer_norm"] = last_layer_grad

            grad_stats["gradients/layer_variance"] = np.var(layer_grad_norms)
            grad_stats["gradients/min_layer_norm"] = min(layer_grad_norms)
            grad_stats["gradients/max_layer_norm"] = max(layer_grad_norms)

            grad_stats["gradients/min_update_ratio"] = min(layer_update_ratios)
            grad_stats["gradients/max_update_ratio"] = max(layer_update_ratios)
            grad_stats["gradients/avg_update_ratio"] = np.mean(layer_update_ratios)

            grad_stats["gradients/vanishing_warning"] = int(first_layer_grad < 1e-5)
            grad_stats["gradients/exploding_warning"] = int(max(layer_grad_norms) > 10.0)

        grad_stats["gradients/total_layers"] = len(layer_grads)
        grad_stats["gradients/global_norm"] = (sum(layer_grad_norms) ** 2) ** 0.5

        return grad_stats

    def _extract_layer_name(self, param_name: str) -> str:
        """Extract layer name from parameter name (e.g., 'conv1.weight' -> 'conv1')."""
        name = param_name.replace('.weight', '').replace('.bias', '')
        name = name.replace('.', '_')
        return name

    def _compute_prediction_metrics(self, model: torch.nn.Module, val_loader, device: str) -> Dict[str, float]:
        """Compute prediction statistics on validation samples."""
        model.eval()
        predictions_logged = 0
        max_predictions = 5

        pred_metrics = {}

        with torch.no_grad():
            for batch in val_loader:
                if predictions_logged >= max_predictions:
                    break

                batch = batch.to(device)

                try:
                    if hasattr(batch, '__getitem__') and 'cell' in batch:
                        x_dict = {'cell': batch['cell'].x}
                        edge_index_dict = {
                            ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
                            ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
                            ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
                        }
                        logits = model(x_dict, edge_index_dict)
                        labels = batch['cell'].y
                    else:
                        logits = model(batch.x, batch.edge_index)
                        labels = batch.y

                    probs = torch.sigmoid(logits)
                    preds = (logits > 0).long()

                    accuracy = (preds == labels).float().mean().item()
                    avg_confidence = probs.mean().item()
                    positive_rate = preds.float().mean().item()

                    pred_metrics.update({
                        f"predictions/sample_{predictions_logged}_accuracy": accuracy,
                        f"predictions/sample_{predictions_logged}_confidence": avg_confidence,
                        f"predictions/sample_{predictions_logged}_positive_rate": positive_rate,
                    })

                    predictions_logged += 1

                except Exception as e:
                    print(f"Warning: Could not compute prediction metrics for batch {predictions_logged}: {e}")
                    predictions_logged += 1
                    continue

        model.train()
        return pred_metrics

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build config_dict based on model_type, including only relevant parameters."""
        model_type = self.config.model.model_type
        
        config_dict = {
            "model_type": model_type,
            "input_dim": self.config.model.input_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "dropout": self.config.model.dropout,
        }
        
        if model_type == "GAT":
            config_dict.update({
                "layer_count": self.config.model.layer_count,
                "gat_heads": self.config.model.gat_heads,
            })
        
        elif model_type == "HeteroGAT":
            config_dict.update({
                "layer_count": self.config.model.layer_count,
                "gat_heads": self.config.model.gat_heads,
                "hgt_heads": self.config.model.hgt_heads,
                "use_batch_norm": self.config.model.use_batch_norm,
            })
        
        elif model_type == "HRM":
            config_dict.update({
                "gat_heads": self.config.model.gat_heads,
                "hgt_heads": self.config.model.hgt_heads,
                "n_cycles": self.config.model.n_cycles,
                "t_micro": self.config.model.t_micro,
                "use_input_injection": self.config.model.use_input_injection,
                "z_dim": self.config.model.z_dim,
                "use_hmod": self.config.model.use_hmod,
                "use_batch_norm": self.config.model.use_batch_norm,
                "same_size_batches": self.config.training.same_size_batches,
            })
        
        elif model_type == "HRM_FullSpatial":
                    config_dict.update({
                        "gat_heads": self.config.model.gat_heads,
                        "hgt_heads": self.config.model.hgt_heads,
                        "hmod_heads": self.config.model.hmod_heads,
                        "n_cycles": self.config.model.n_cycles,
                        "t_micro": self.config.model.t_micro,
                        "learning_rate": self.config.training.learning_rate,
                        "weight_decay": self.config.training.weight_decay,
                        "focal_alpha": self.config.training.focal_alpha,
                        "focal_gamma": self.config.training.focal_gamma,
                        "cosine_t_max": self.config.training.cosine_t_max,
                        "cosine_eta_min": self.config.training.cosine_eta_min,
                        "warmup_epochs": getattr(self.config.training, 'warmup_epochs', 0),
                        "warmup_start_factor": getattr(self.config.training, 'warmup_start_factor', 0.1),
                    })
        
        elif model_type == "GNN":
            config_dict.update({
                "layer_count": self.config.model.layer_count,
            })
        
        else:
            config_dict.update({
                "layer_count": getattr(self.config.model, 'layer_count', None),
                "gat_heads": getattr(self.config.model, 'gat_heads', None),
                "hgt_heads": getattr(self.config.model, 'hgt_heads', None),
                "n_cycles": getattr(self.config.model, 'n_cycles', None),
                "t_micro": getattr(self.config.model, 't_micro', None),
                "use_input_injection": getattr(self.config.model, 'use_input_injection', None),
                "z_dim": getattr(self.config.model, 'z_dim', None),
                "use_hmod": getattr(self.config.model, 'use_hmod', None),
                "use_batch_norm": getattr(self.config.model, 'use_batch_norm', None),
                "same_size_batches": getattr(self.config.training, 'same_size_batches', None),
                "hmod_heads": getattr(self.config.model, 'hmod_heads', None),
            })
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        return config_dict

    def save_checkpoint(self, model: torch.nn.Module, optimizer, epoch: int,
                    metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config_dict': self._build_config_dict(),
        }

        if epoch % self.config.experiment.save_model_every_n_epochs == 0:
            checkpoint_path = Path(self.config.experiment.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = Path(self.config.experiment.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

            wandb.log({
                "best_model/epoch": epoch,
                "best_model/saved": 1,
            }, step=epoch)

    def set_current_lr(self, lr: float):
        """Set current learning rate for logging."""
        self._current_lr = lr

    def finish(self):
        """Clean up and finish experiment."""
        total_time = time.time() - self.start_time
        final_metrics = {
            "experiment/total_time_minutes": total_time / 60,
            "experiment/peak_memory_mb": self.peak_memory,
        }

        wandb.log(final_metrics, step=9999)
        wandb.finish()
        print("Experiment tracking finished")

def create_wandb_sweep(config: Dict[str, Any], project_name: str) -> str:
    """Create a wandb hyperparameter sweep and return sweep_id."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val/f1',
            'goal': 'maximize'
        },
        'parameters': config
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    return sweep_id


EXAMPLE_SWEEP_CONFIG = {
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-2
    },
    'hidden_dim': {
        'values': [128, 256, 512]
    },
    'dropout': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.5
    },
    'focal_gamma': {
        'values': [1.0, 2.0, 3.0, 5.0]
    }
}