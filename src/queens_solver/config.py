import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from queens_solver.data.dataset import SizeBucketBatchSampler

logger = logging.getLogger(__name__)

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
    dropout: float = 0.12
    use_batch_norm: bool = False

    gat_heads: int = 4
    hgt_heads: int = 4

    model_type: str = "HRM"  # "GAT", "HeteroGAT", or "HRM"
    hetero_aggr: str = "sum"

    # HRM-specific
    n_cycles: int = 3
    t_micro: int = 2
    hmod_heads: int = 4
    use_input_injection: bool = True
    z_dim: int = 128
    use_hmod: bool = False # When true make same size batches True as well

    input_injection_layers: Optional[list[int]] = field(default_factory=lambda: [2, 5])

class BenchmarkConfig:
    model_type: str = "hrm"
    input_dim: int = 14
    hidden_dim: int = 128
    layers: int = 6
    dropout: float = 0.12
    n_heads = 4
    microsteps = 2
    n_cycles = 3

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 24
    batch_size: int = 512 
    learning_rate: float = 1.5e-3
    weight_decay: float = 0.000003
    val_ratio: float = 0.10
    warmup_epochs: int = 0 
    warmup_start_factor: float = 0.1

    # Dataset combination
    combine_state0: bool = True  # Combine state-0 into training set upfront
    state0_json_path: str = "data/State0TrainingSet.json"

    # Batch sampler options
    same_size_batches: bool = True 
    drop_last: bool = True  # Drop last incomplete batch
    
    # Legacy curriculum options (unused when combine_state0=True)
    state0_epochs: list = field(default_factory=lambda: [])
    lr_reduce_epoch: list = field(default_factory=lambda: [])
    lr_reduce_factor: float = 0.5
    mixed_ratio: float = 0.75

    focal_alpha: float = 0.37
    focal_gamma: float = 2.2

    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "plateau", "step", "none"
    cosine_t_max: int = 24
    cosine_eta_min: float = 1e-6
    constant_lr_epochs = 0
    constant_lr = 1e-05

@dataclass
class DataConfig:
    """Data loading configuration."""
    train_json: str = "data/StateTrainingSet.json"
    auto_reg_json: str = "data/StateValSet.json"
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

    checkpoint_dir: str = 'checkpoints/test/HRM' #"checkpoints/transformer/HRM/FullSpatial"
    log_dir: str = "logs"

@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    compile_model: bool = False
    profile_memory: bool = False

class Config:
    """Main configuration combining all sub-configs."""

    def __init__(self, **kwargs):
        self.model = ModelConfig()
        self.benchmark = BenchmarkConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.system = SystemConfig()

        self.update_from_dict(kwargs)

        logger.debug(f"DataLoader workers: {self.data.num_workers}")

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
            "benchmark": self.benchmark.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__,
            "system": self.system.__dict__,
        }

BASELINE_CONFIG = {
    "experiment": {
        "experiment_name": "New Repo HRM ",
        "tags": ["Testing", 'Graph HRM'],
        "notes": "Testing HRM model in new repository."
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
