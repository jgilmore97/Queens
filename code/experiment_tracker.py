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
from torch_geometric.data import Batch

class MemoryEfficientLogger:
    """Memory-conscious logging with buffering and sampling."""
    
    def __init__(self, config):
        self.config = config.experiment
        self.buffer_size = 100
        self.metric_buffer = defaultdict(deque)
        self.sample_pool = []
        self.logged_samples = 0
        
    def add_scalar(self, key: str, value: float, step: Optional[int] = None):
        """Add scalar metric to buffer."""
        self.metric_buffer[key].append((value, step, time.time()))
        
        # Prevent buffer overflow
        if len(self.metric_buffer[key]) > self.buffer_size:
            self.metric_buffer[key].popleft()
    
    def add_sample(self, sample_data: Dict[str, Any]):
        """Add training sample for later logging (with memory limits)."""
        if self.logged_samples < self.config.max_logged_samples:
            self.sample_pool.append(sample_data)
            if len(self.sample_pool) > self.config.max_logged_samples:
                # Replace random sample to maintain diversity
                idx = random.randint(0, len(self.sample_pool) - 1)
                self.sample_pool[idx] = sample_data
    
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

class ExperimentTracker:
    """Main experiment tracking class with memory management."""
    
    def __init__(self, config, resume_id: Optional[str] = None):
        self.config = config
        self.logger = MemoryEfficientLogger(config)
        self.start_time = time.time()
        self.step_count = 0
        self.epoch_count = 0
        
        # Initialize wandb
        self._init_wandb(resume_id)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory = 0
        
        # Create directories
        Path(config.experiment.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.experiment.log_dir).mkdir(exist_ok=True)
    
    def _init_wandb(self, resume_id: Optional[str]):
        """Initialize Weights & Biases with configuration."""
        wandb.init(
            project=self.config.experiment.project_name,
            name=self.config.experiment.experiment_name,
            tags=self.config.experiment.tags or [],
            notes=self.config.experiment.notes,
            config=self.config.to_dict(),
            resume="allow" if resume_id else None,
            id=resume_id,
        )
        
        # Log system info
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system and environment information."""
        system_info = {
            "system/device": self.config.system.device,
            "system/cuda_available": torch.cuda.is_available(),
            "system/cpu_count": psutil.cpu_count(),
            "system/memory_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "system/gpu_name": torch.cuda.get_device_name(),
                "system/gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            })
        
        wandb.log(system_info)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log metrics with optional prefix."""
        if step is None:
            step = self.step_count
            
        prefixed_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        
        for key, value in prefixed_metrics.items():
            self.logger.add_scalar(key, value, step)
        
        # Flush periodically
        if step % self.config.experiment.log_every_n_batches == 0:
            self.logger.flush(step)
    
    def log_model_performance(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log comprehensive model performance metrics."""
        all_metrics = {}
        
        # Add prefixes and combine
        for k, v in train_metrics.items():
            all_metrics[f"train/{k}"] = v
        for k, v in val_metrics.items():
            all_metrics[f"val/{k}"] = v
        
        # Add epoch info
        all_metrics["epoch"] = epoch
        all_metrics["learning_rate"] = self._get_current_lr()
        
        # Memory usage
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, memory_mb)
        all_metrics["system/memory_mb"] = memory_mb
        all_metrics["system/peak_memory_mb"] = self.peak_memory
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            all_metrics["system/gpu_memory_mb"] = gpu_memory_mb
        
        wandb.log(all_metrics, step=epoch)
    
    def log_gradients(self, model: torch.nn.Module, epoch: int):
        """Log gradient statistics (memory efficient)."""
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
                
                # Log statistics for major layers only (memory efficient)
                if any(layer_type in name for layer_type in ['conv', 'linear', 'attention']):
                    grad_stats[f"gradients/{name}_norm"] = grad_norm
                    grad_stats[f"gradients/{name}_mean"] = param.grad.mean().item()
                    grad_stats[f"gradients/{name}_std"] = param.grad.std().item()
        
        grad_stats["gradients/total_norm"] = total_norm ** 0.5
        grad_stats["gradients/param_count"] = param_count
        
        wandb.log(grad_stats, step=epoch)
    
    def log_predictions(self, model: torch.nn.Module, val_loader, epoch: int, device: str):
        """Log sample predictions for analysis."""
        if epoch % self.config.experiment.log_predictions_every_n_epochs != 0:
            return
        
        model.eval()
        predictions_logged = 0
        max_predictions = 10  # Limit for memory
        
        with torch.no_grad():
            for batch in val_loader:
                if predictions_logged >= max_predictions:
                    break
                    
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(logits)
                preds = (logits > 0).long()
                
                # Log a few examples from this batch
                batch_size = batch.num_graphs
                for i in range(min(3, batch_size)):  # Max 3 per batch
                    if predictions_logged >= max_predictions:
                        break
                    
                    # Extract single graph data
                    graph_mask = batch.batch == i
                    graph_probs = probs[graph_mask].cpu().numpy()
                    graph_preds = preds[graph_mask].cpu().numpy()
                    graph_labels = batch.y[graph_mask].cpu().numpy()
                    
                    # Calculate metrics for this graph
                    correct = (graph_preds == graph_labels).sum()
                    total = len(graph_labels)
                    accuracy = correct / total
                    
                    # Log prediction summary
                    wandb.log({
                        f"predictions/sample_{predictions_logged}_accuracy": accuracy,
                        f"predictions/sample_{predictions_logged}_avg_confidence": graph_probs.mean(),
                        f"predictions/sample_{predictions_logged}_positive_preds": graph_preds.sum(),
                        f"predictions/sample_{predictions_logged}_positive_labels": graph_labels.sum(),
                    }, step=epoch)
                    
                    predictions_logged += 1
        
        model.train()
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer, epoch: int, 
                       metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
        }
        
        # Save regular checkpoint
        if epoch % self.config.experiment.save_model_every_n_epochs == 0:
            checkpoint_path = Path(self.config.experiment.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.experiment.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Log model artifact to wandb
            artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
            artifact.add_file(str(best_path))
            wandb.log_artifact(artifact)
    
    def _get_current_lr(self) -> float:
        """Get current learning rate (implement based on your optimizer)."""
        # This will be set by the training loop
        return getattr(self, '_current_lr', 0.0)
    
    def set_current_lr(self, lr: float):
        """Set current learning rate for logging."""
        self._current_lr = lr
    
    def log_hyperparameter_sweep(self, sweep_config: Dict[str, Any]):
        """Log hyperparameter sweep configuration."""
        wandb.log({"sweep_config": sweep_config})
    
    def finish(self):
        """Clean up and finish experiment."""
        # Final flush
        self.logger.flush(self.step_count)
        
        # Log final statistics
        total_time = time.time() - self.start_time
        wandb.log({
            "experiment/total_time_minutes": total_time / 60,
            "experiment/peak_memory_mb": self.peak_memory,
        })
        
        wandb.finish()

def create_wandb_sweep(config: Dict[str, Any], project_name: str) -> str:
    """Create a wandb hyperparameter sweep."""
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val/f1',
            'goal': 'maximize'
        },
        'parameters': config
    }
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    return sweep_id

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