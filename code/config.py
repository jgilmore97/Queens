from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 14
    hidden_dim: int = 256
    layer_count: int = 3
    dropout: float = 0.2
    heads: int = 2
    model_type: str = "GAT"  # "GAT" or "GNN"

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.10
    
    # Loss function
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6

@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    train_json: str = "10k_training_set_with_states.json"
    test_json: str = "test_set_with_states.json"
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    seed: int = 42

@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""
    project_name: str = "queens-puzzle-ml"
    experiment_name: Optional[str] = 'test'
    entity: Optional[str] = None  # Add W&B entity field
    tags: list = 'test'
    notes: str = ""
    
    # Logging frequencies (removed batch-level settings)
    log_gradients_every_n_epochs: int = 5
    log_predictions_every_n_epochs: int = 2
    save_model_every_n_epochs: int = 10
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

@dataclass
class SystemConfig:
    """System and performance configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    profile_memory: bool = False

class Config:
    """Main configuration class combining all sub-configs."""
    
    def __init__(self, **kwargs):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.system = SystemConfig()
        
        # Override with any provided kwargs
        self.update_from_dict(kwargs)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__,
            "system": self.system.__dict__,
        }

# Example configurations for different experiment types
BASELINE_CONFIG = {
    "experiment": {
        "experiment_name": "baseline_gat",
        "tags": ["baseline", "gat"],
        "notes": "Baseline GAT model with current architecture"
    }
}

ABLATION_CONFIG = {
    "experiment": {
        "experiment_name": "architecture_ablation",
        "tags": ["ablation", "architecture"],
        "notes": "Testing different model architectures"
    }
}

HYPEROPT_CONFIG = {
    "experiment": {
        "experiment_name": "hyperparameter_sweep",
        "tags": ["sweep", "optimization"],
        "notes": "Systematic hyperparameter optimization"
    },
    "training": {
        "epochs": 15  # Shorter runs for sweeps
    }
}