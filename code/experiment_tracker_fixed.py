"""
Simplified experiment tracking with W&B - no artifacts, no bucket uploads.
Memory-efficient logging with basic functionality only.
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

class MemoryEfficientLogger:
    """Memory-conscious logging with buffering and sampling."""
    
    def __init__(self, config):
        self.config = config.experiment
        self.buffer_size = 100
        self.metric_buffer = defaultdict(deque)
        
    def add_scalar(self, key: str, value: float, step: Optional[int] = None):
        """Add scalar metric to buffer."""
        self.metric_buffer[key].append((value, step, time.time()))
        
        # Prevent buffer overflow
        if len(self.metric_buffer[key]) > self.buffer_size:
            self.metric_buffer[key].popleft()
    
    def flush(self, step: int):
        """Flush buffered metrics to wandb."""
        if not self.metric_buffer:
            return
            
        # Aggregate buffered metrics
        log_dict = {}
        for key, values in self.metric_buffer.items():
            if values:
                recent_values = [v[0] for v in values]
                log_dict[key] = np.mean(recent_values)
        
        if log_dict:
            wandb.log(log_dict, step=step)
        
        # Clear buffers
        self.metric_buffer.clear()

"""
Simplified experiment tracking with single step counter for epoch-level logging only.
"""

class ExperimentTracker:
    """
    Simplified experiment tracking with single step counter for epoch-level metrics only.
    No more batch-level logging to avoid W&B step conflicts.
    """
    
    def __init__(self, config, resume_id: Optional[str] = None):
        self.config = config
        self.start_time = time.time()
        
        # Single step counter for all logging
        self.step = 0
        
        # Initialize wandb
        self._init_wandb(resume_id)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory = 0
        
        # Create directories
        Path(config.experiment.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.experiment.log_dir).mkdir(exist_ok=True)
    
    def _init_wandb(self, resume_id: Optional[str]):
        """Initialize W&B with simple step metric."""
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
        
        # Log basic system info
        self._log_system_info()
    
    def _log_system_info(self):
        """Log basic system information."""
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
        
        wandb.log(system_metrics, step=self.step)
        self.step += 1
    
    def log_model_performance(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log epoch-level metrics with single step counter."""
        all_metrics = {}
        
        # Add prefixes
        for k, v in train_metrics.items():
            all_metrics[f"train/{k}"] = v
        for k, v in val_metrics.items():
            all_metrics[f"val/{k}"] = v
        
        # Add metadata
        all_metrics["epoch"] = epoch
        all_metrics["learning_rate"] = self._get_current_lr()
        
        # System metrics
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
        
        # Log everything with single step counter
        wandb.log(all_metrics, step=self.step)
        self.step += 1
    
    def log_gradients(self, model: torch.nn.Module, epoch: int):
        """Log gradient statistics."""
        if epoch % self.config.experiment.log_gradients_every_n_epochs != 0:
            return
            
        grad_stats = {}
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                param_count += param.numel()
                
                if any(layer_type in name for layer_type in ['conv', 'linear', 'attention']):
                    grad_stats[f"gradients/{name}_norm"] = grad_norm
        
        grad_stats["gradients/total_norm"] = total_norm ** 0.5
        grad_stats["gradients/param_count"] = param_count
        
        wandb.log(grad_stats, step=self.step)
        self.step += 1
    
    def log_predictions(self, model: torch.nn.Module, val_loader, epoch: int, device: str):
        """Log sample predictions."""
        if epoch % self.config.experiment.log_predictions_every_n_epochs != 0:
            return
        
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
        
        if pred_metrics:
            wandb.log(pred_metrics, step=self.step)
            self.step += 1
        
        model.train()
    
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
            
            # Log that we saved the best model
            wandb.log({
                "best_model/epoch": epoch, 
                "best_model/saved": 1,
            }, step=self.step)
            self.step += 1
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        return getattr(self, '_current_lr', 0.0)
    
    def set_current_lr(self, lr: float):
        """Set current learning rate for logging."""
        self._current_lr = lr
    
    def finish(self):
        """Clean up and finish experiment."""
        total_time = time.time() - self.start_time
        final_metrics = {
            "experiment/total_time_minutes": total_time / 60,
            "experiment/peak_memory_mb": self.peak_memory,
            "experiment/total_steps": self.step,
        }
        
        wandb.log(final_metrics, step=self.step)
        wandb.finish()
        print("🏁 Experiment tracking finished")

# Example sweep configuration
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