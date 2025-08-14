from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import os

def _detect_notebook_environment():
    """Detect if running in a notebook environment where multiprocessing can be problematic."""
    try:
        # Check for common notebook indicators
        if 'COLAB_GPU' in os.environ:
            return True
        if 'JPY_PARENT_PID' in os.environ:
            return True
        # Check if IPython is available and we're in a notebook
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return 'google.colab' in str(get_ipython())
        except ImportError:
            pass
    except:
        pass
    return False

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 14
    hidden_dim: int = 128
    layer_count: int = 6
    dropout: float = 0.2
    heads: int = 2
    input_injection_layers: Optional[list[int]] = [2,5]  # Layers to inject input features into
    model_type: str = "HeteroGAT"  # "GAT", "HeteroGAT", or "GNN"
    
    # Heterogeneous model specific settings
    hetero_aggr: str = "sum"  # How to aggregate messages from different edge types: "sum", "mean", "max"

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.10
    
    # Loss function
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Scheduler
    scheduler_type: str = "plateau"  # "cosine", "step", "plateau"
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6

@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    train_json: str = "10k_training_set_with_states.json"
    test_json: str =  "test_set_with_states.json" #"FinalTestOfficialPuzzles.json" -use for small set of never mutated official LI boards
    num_workers: int = 0 if _detect_notebook_environment() else 4  # Auto-detect and disable for notebooks
    pin_memory: bool = True
    shuffle_train: bool = True
    seed: int = 42

@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""
    project_name: str = "queens-puzzle-ml"
    experiment_name: Optional[str] = 'test'
    tags: list = 'test'
    notes: str = ""
    
    # Logging frequencies
    log_gradients_every_n_epochs: int = 1
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
    compile_model: bool = False
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
        
        # Print multiprocessing status for debugging
        if self.data.num_workers == 0:
            print("ℹ️  DataLoader multiprocessing disabled (prevents worker shutdown errors)")
        else:
            print(f"ℹ️  DataLoader using {self.data.num_workers} worker processes")
    
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
        "experiment_name": "thinner_deeper_net",
        "tags": ["heterogat", "thin_deep"],
        "notes": "Baseline heterogeneous GAT with thinner and deeper architecture"
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
        "epochs": 15
    }
}