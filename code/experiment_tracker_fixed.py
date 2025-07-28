"""
Fixed experiment tracking with W&B - consolidated epoch-level logging only.
No more step conflicts or batch-level tracking.
"""

import os
import time
import psutil
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

import torch
import numpy as np
import wandb

class ExperimentTracker:
    """
    Fixed experiment tracking with consolidated epoch-level logging only.
    Uses epoch number as the W&B step to avoid step conflicts.
    """
    
    def __init__(self, config, resume_id: Optional[str] = None):
        self.config = config
        self.start_time = time.time()
        
        # Initialize wandb
        self._init_wandb(resume_id)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory = 0
        
        # Current learning rate tracking
        self._current_lr = config.training.learning_rate
        
        # Create directories
        Path(config.experiment.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.experiment.log_dir).mkdir(exist_ok=True)
        
        print("✅ Experiment tracker initialized with consolidated logging")
    
    def _init_wandb(self, resume_id: Optional[str]):
        """Initialize W&B with simple config."""
        # Simple config dict
        simple_config = {
            "model_type": self.config.model.model_type,
            "input_dim": self.config.model.input_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "layer_count": self.config.model.layer_count,
            "dropout": self.config.model.dropout,
            "heads": self.config.model.heads,
            "epochs": self.config.training.epochs,
            "batch_size": self.config.training.batch_size,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "focal_alpha": self.config.training.focal_alpha,
            "focal_gamma": self.config.training.focal_gamma,
            "device": self.config.system.device,
        }
        
        # Build wandb.init args
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
        
        # Log system info once at the beginning (step 0)
        self._log_system_info()
    
    def _log_system_info(self):
        """Log basic system information at step 0."""
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
        
        # Log system info at step 0
        wandb.log(system_metrics, step=0)
    
    def log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int, 
                         model: Optional[torch.nn.Module] = None, val_loader=None, device: str = None):
        """
        Consolidated logging for all epoch-level metrics.
        Uses epoch number as W&B step to avoid conflicts.
        """
        # Start building the consolidated metrics dict
        all_metrics = {}
        
        # Add train/val metrics with prefixes
        for k, v in train_metrics.items():
            all_metrics[f"train/{k}"] = v
        for k, v in val_metrics.items():
            all_metrics[f"val/{k}"] = v
        
        # Add epoch metadata
        all_metrics["epoch"] = epoch
        all_metrics["learning_rate"] = self._current_lr
        
        # Add system metrics
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
        
        # Add gradient metrics if model provided and it's time to log them
        if (model is not None and 
            epoch % self.config.experiment.log_gradients_every_n_epochs == 0):
            grad_metrics = self._compute_gradient_metrics(model)
            all_metrics.update(grad_metrics)
        
        # Add prediction metrics if val_loader provided and it's time to log them
        if (model is not None and val_loader is not None and device is not None and
            epoch % self.config.experiment.log_predictions_every_n_epochs == 0):
            pred_metrics = self._compute_prediction_metrics(model, val_loader, device)
            all_metrics.update(pred_metrics)
        
        # Single consolidated log call using epoch as step
        wandb.log(all_metrics, step=epoch)
    
    def _compute_gradient_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute proper vanishing gradient analysis.
        Tracks layer-wise gradients, ratios, and gradient flow.
        """
        grad_stats = {}
        
        # Collect layer-wise gradient norms
        layer_grads = {}
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # Extract layer identifier (e.g., "conv1", "convs.0", "linear")
                layer_name = self._extract_layer_name(name)
                
                if layer_name not in layer_grads:
                    layer_grads[layer_name] = {
                        'grad_norm': 0.0,
                        'param_norm': 0.0,
                        'param_count': 0
                    }
                
                # Accumulate for this layer
                layer_grads[layer_name]['grad_norm'] += grad_norm ** 2
                layer_grads[layer_name]['param_norm'] += param_norm ** 2
                layer_grads[layer_name]['param_count'] += param.numel()
                
                layer_names.append(layer_name)
        
        # Calculate layer-wise statistics
        layer_grad_norms = []
        layer_update_ratios = []
        
        for layer_name, stats in layer_grads.items():
            # Final L2 norm for this layer
            layer_grad_norm = (stats['grad_norm'] ** 0.5)
            layer_param_norm = (stats['param_norm'] ** 0.5)
            
            # Update ratio: how much this layer will change relative to its current values
            update_ratio = layer_grad_norm / (layer_param_norm + 1e-8)
            
            # Store individual layer metrics
            grad_stats[f"gradients/layer_{layer_name}_norm"] = layer_grad_norm
            grad_stats[f"gradients/layer_{layer_name}_update_ratio"] = update_ratio
            grad_stats[f"gradients/layer_{layer_name}_param_count"] = stats['param_count']
            
            layer_grad_norms.append(layer_grad_norm)
            layer_update_ratios.append(update_ratio)
        
        if len(layer_grad_norms) >= 2:
            # Gradient flow analysis
            first_layer_grad = layer_grad_norms[0]
            last_layer_grad = layer_grad_norms[-1]
            
            # Vanishing gradient indicators
            grad_stats["gradients/first_last_ratio"] = first_layer_grad / (last_layer_grad + 1e-8)
            grad_stats["gradients/first_layer_norm"] = first_layer_grad
            grad_stats["gradients/last_layer_norm"] = last_layer_grad
            
            # Gradient variance across layers (higher = more uneven flow)
            grad_stats["gradients/layer_variance"] = np.var(layer_grad_norms)
            grad_stats["gradients/min_layer_norm"] = min(layer_grad_norms)
            grad_stats["gradients/max_layer_norm"] = max(layer_grad_norms)
            
            # Update ratio analysis
            grad_stats["gradients/min_update_ratio"] = min(layer_update_ratios)
            grad_stats["gradients/max_update_ratio"] = max(layer_update_ratios)
            grad_stats["gradients/avg_update_ratio"] = np.mean(layer_update_ratios)
            
            # Vanishing gradient warning flags
            grad_stats["gradients/vanishing_warning"] = int(first_layer_grad < 1e-5)
            grad_stats["gradients/exploding_warning"] = int(max(layer_grad_norms) > 10.0)
        
        # Overall statistics
        grad_stats["gradients/total_layers"] = len(layer_grads)
        grad_stats["gradients/global_norm"] = (sum(layer_grad_norms) ** 2) ** 0.5
        
        return grad_stats
    
    def _extract_layer_name(self, param_name: str) -> str:
        """
        Extract meaningful layer name from parameter name.
        Examples:
            'conv1.weight' -> 'conv1'
            'convs.0.weight' -> 'convs_0'
            'linear.bias' -> 'linear'
        """
        # Remove .weight, .bias suffixes
        name = param_name.replace('.weight', '').replace('.bias', '')
        
        # Handle module lists (convs.0 -> convs_0)
        name = name.replace('.', '_')
        
        return name
    
    def _compute_prediction_metrics(self, model: torch.nn.Module, val_loader, device: str) -> Dict[str, float]:
        """Compute prediction statistics and return as dict."""
        model.eval()
        predictions_logged = 0
        max_predictions = 5
        
        pred_metrics = {}
        
        with torch.no_grad():
            for batch in val_loader:
                if predictions_logged >= max_predictions:
                    break
                    
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(logits)
                preds = (logits > 0).long()
                
                accuracy = (preds == batch.y).float().mean().item()
                avg_confidence = probs.mean().item()
                positive_rate = preds.float().mean().item()
                
                pred_metrics.update({
                    f"predictions/sample_{predictions_logged}_accuracy": accuracy,
                    f"predictions/sample_{predictions_logged}_confidence": avg_confidence,
                    f"predictions/sample_{predictions_logged}_positive_rate": positive_rate,
                })
                
                predictions_logged += 1
        
        model.train()
        return pred_metrics
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer, epoch: int, 
                       metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint locally."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config_dict': {
                "model_type": self.config.model.model_type,
                "input_dim": self.config.model.input_dim,
                "hidden_dim": self.config.model.hidden_dim,
                "layer_count": self.config.model.layer_count,
                "dropout": self.config.model.dropout,
                "heads": self.config.model.heads,
            }
        }
        
        if epoch % self.config.experiment.save_model_every_n_epochs == 0:
            checkpoint_path = Path(self.config.experiment.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = Path(self.config.experiment.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path}")
            
            # Log that we saved the best model (will be included in next epoch log)
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
        
        # Use a high step number for final metrics to avoid conflicts
        wandb.log(final_metrics, step=9999)
        wandb.finish()
        print("🏁 Experiment tracking finished")

def create_wandb_sweep(config: Dict[str, Any], project_name: str) -> str:
    """Create a wandb hyperparameter sweep."""
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