from dataclasses import dataclass, field
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
    layer_count: int = 6  # Used by GAT/HeteroGAT only
    dropout: float = 0.2
    
    # Separate head configurations
    gat_heads: int = 2
    hgt_heads: int = 4
    
    # Model selection
    model_type: str = "HRM"  # "GAT", "HeteroGAT", or "HRM"
    hetero_aggr: str = "sum"
    
    # HRM-specific parameters
    n_cycles: int = 3              # Number of H-module updates
    t_micro: int = 2               # L-module micro-steps per cycle
    use_input_injection: bool = True
    z_init: str = "zeros"          # "zeros" or "learned"
    h_pooling_heads: int = 4  # Number of attention heads for H-module pooling
    
    input_injection_layers: Optional[list[int]] = field(default_factory=lambda: [2, 5])

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 18
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.10
    
    # Dataset switching
    switch_epoch: int = 5  # Epoch to switch to state-0 dataset (999 = never switch)
    state0_json_path: str = "state0_training_states.json"
    mixed_ratio: float = 0.75  # Ratio of state-0 to old data in mixed dataset
    
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
    checkpoint_dir: str = "checkpoints/transformer/HRM"
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
        "experiment_name": "RUN 3 of HRM inspired Model - add attention based h pooling, increase cycles",
        "tags": ["HRM", "post-train", "attention pooling"],
        "notes": "HRM with attention-based H-module pooling"
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