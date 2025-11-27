from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import os

def _detect_notebook_environment():
    """Detect if running in a notebook environment."""
    try:
        if 'COLAB_GPU' in os.environ:
            return True
        if 'JPY_PARENT_PID' in os.environ:
            return True
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

    gat_heads: int = 2
    hgt_heads: int = 4

    model_type: str = "HRM"  # "GAT", "HeteroGAT", or "HRM"
    hetero_aggr: str = "sum"

    # HRM-specific
    n_cycles: int = 3
    t_micro: int = 2
    use_input_injection: bool = True
    z_init: str = "zeros"
    h_pooling_heads: int = 4

    input_injection_layers: Optional[list[int]] = field(default_factory=lambda: [2, 5])

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 18
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.10

    switch_epoch: int = 5  # Epoch to switch to state-0 dataset (999 = never)
    state0_json_path: str = "state0_training_states.json"
    mixed_ratio: float = 0.75

    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    scheduler_type: str = "plateau"
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6

@dataclass
class DataConfig:
    """Data loading configuration."""
    train_json: str = "10k_training_set_with_states.json"
    test_json: str = "test_set_with_states.json"
    num_workers: int = 0 if _detect_notebook_environment() else 4
    pin_memory: bool = True
    shuffle_train: bool = True
    seed: int = 42

@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    project_name: str = "queens-puzzle-ml"
    experiment_name: Optional[str] = 'test'
    tags: list = 'test'
    notes: str = ""

    log_gradients_every_n_epochs: int = 1
    log_predictions_every_n_epochs: int = 2
    save_model_every_n_epochs: int = 10

    checkpoint_dir: str = "checkpoints/transformer/HRM"
    log_dir: str = "logs"

@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False
    profile_memory: bool = False

class Config:
    """Main configuration combining all sub-configs."""

    def __init__(self, **kwargs):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.system = SystemConfig()

        self.update_from_dict(kwargs)

        if self.data.num_workers == 0:
            print("DataLoader multiprocessing disabled")
        else:
            print(f"DataLoader using {self.data.num_workers} workers")

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__,
            "system": self.system.__dict__,
        }

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

# HRM-specific hyperparameter tuning configuration
HRM_TUNING_SPACE = {
    # Architecture hyperparameters
    "t_micro": {"type": "categorical", "values": [2, 3]},  # Removed 4 (OOM risk)
    "hidden_dim": {"type": "categorical", "values": [128, 192, 256]},
    "gat_heads": {"type": "categorical", "values": [2, 4]},
    "hgt_heads": {"type": "categorical", "values": [4, 8]},  # Removed 6 (not divisible by 128/256)
    "h_pooling_heads": {"type": "categorical", "values": [2, 4, 8]},
    "dropout": {"type": "categorical", "values": [0.1, 0.2, 0.3]},

    # Training hyperparameters
    "learning_rate": {"type": "loguniform", "low": 3e-4, "high": 3e-3},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-4},

    # Fixed hyperparameters (not tuned)
    "fixed": {
        "n_cycles": 3,
        "use_input_injection": True,
        "z_init": "zeros",
        "input_dim": 14,
    }
}

HRM_TUNING_CONFIG = {
    # Trial settings
    "n_trials": 40,
    "epochs_per_trial": 5,
    "full_training_epochs": 20,

    # Data settings - use state0-heavy from start for full-solve optimization
    "use_mixed_from_start": True,
    "mixed_ratio": 0.75,  # 75% state-0, 25% multi-state

    # Evaluation settings
    "eval_every_n_epochs": 5,  # Only eval at end of trial (faster)
    "batch_size": 256,

    # Pruning settings
    "pruning_warmup_epochs": 3,  # Don't prune before this epoch

    # Paths (consistent with train_all_models.py)
    "state0_json_path": "data/State0TrainingSet.json",
    "multistate_json_path": "data/StateTrainingSet.json",
    "full_solve_val_path": "data/StateValSet.json",
    "results_dir": "tuning_results",
}
