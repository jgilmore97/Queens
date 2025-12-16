import os
import torch
from pathlib import Path

from train import run_training_with_tracking_hetero, run_training_with_tracking
from model import GAT, HeteroGAT, HRM
from data_loader import get_queens_loaders, get_benchmark_loaders, get_combined_queens_loaders
from config import Config, BASELINE_CONFIG, BenchmarkConfig
from bm_model import BenchmarkComparisonModel
from bm_train import benchmark_training

import random
import numpy as np

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main_heterogeneous_training():
    """Train heterogeneous GAT model with experiment tracking."""

    set_seed(42)

    hetero_config = Config(**BASELINE_CONFIG)

    print("=== Queens Puzzle ML Training - HETEROGENEOUS ===")
    print(f"Device: {hetero_config.system.device}")
    print(f"Experiment: {hetero_config.experiment.experiment_name}")
    print("Using heterogeneous edges with constraint-specific attention")

    print("\nLoading datasets...")
    train_loader, val_loader = get_queens_loaders(
        hetero_config.data.train_json,
        batch_size=hetero_config.training.batch_size,
        val_ratio=hetero_config.training.val_ratio,
        seed=hetero_config.data.seed,
        num_workers=hetero_config.data.num_workers,
        pin_memory=hetero_config.data.pin_memory,
        shuffle_train=hetero_config.data.shuffle_train,
    )

    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print("Test set reserved for final evaluation only")

    print(f"\nCreating HETEROGENEOUS {hetero_config.model.model_type} model...")
    print("Edge types: line_constraint, region_constraint, diagonal_constraint")

    model = HeteroGAT(
        input_dim=hetero_config.model.input_dim,
        hidden_dim=hetero_config.model.hidden_dim,
        layer_count=hetero_config.model.layer_count,
        dropout=hetero_config.model.dropout,
        gat_heads=hetero_config.model.gat_heads,
        hgt_heads=hetero_config.model.hgt_heads
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"\nStarting HETEROGENEOUS training for {hetero_config.training.epochs} epochs...")

    try:
        model, best_f1 = run_training_with_tracking_hetero(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=hetero_config
        )

        print(f"\nHETEROGENEOUS training completed! Best validation F1: {best_f1:.4f}")
        print("Each constraint type learned specialized attention patterns")
        print("Model checkpoints saved for future test evaluation")

        return model, best_f1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, 0.0
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

def main_hrm_training():
    """Train HRM model with experiment tracking."""

    set_seed(42)

    hrm_config = Config(**BASELINE_CONFIG)

    print("=== Queens Puzzle ML Training - HRM ===")
    print(f"Device: {hrm_config.system.device}")
    print(f"Experiment: {hrm_config.experiment.experiment_name}")
    print("Using Hierarchical Reasoning Model (HRM)")

    print("\nLoading datasets...")
    
    if hrm_config.training.combine_state0:
        train_loader, val_loader = get_combined_queens_loaders(
            hrm_config.data.train_json,
            hrm_config.training.state0_json_path,
            batch_size=hrm_config.training.batch_size,
            val_ratio=hrm_config.training.val_ratio,
            seed=hrm_config.data.seed,
            num_workers=hrm_config.data.num_workers,
            pin_memory=hrm_config.data.pin_memory,
            shuffle_train=hrm_config.data.shuffle_train,
        )
    else:
        train_loader, val_loader = get_queens_loaders(
            hrm_config.data.train_json,
            batch_size=hrm_config.training.batch_size,
            val_ratio=hrm_config.training.val_ratio,
            seed=hrm_config.data.seed,
            num_workers=hrm_config.data.num_workers,
            pin_memory=hrm_config.data.pin_memory,
            shuffle_train=hrm_config.data.shuffle_train,
        )

    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")

    print(f"\nCreating HRM model...")
    model = HRM(
        input_dim=hrm_config.model.input_dim,
        hidden_dim=hrm_config.model.hidden_dim,
        gat_heads=hrm_config.model.gat_heads,
        hgt_heads=hrm_config.model.hgt_heads,
        dropout=hrm_config.model.dropout,
        use_batch_norm=True,
        n_cycles=hrm_config.model.n_cycles,
        t_micro=hrm_config.model.t_micro,
        use_input_injection=hrm_config.model.use_input_injection,
        z_dim=hrm_config.model.z_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"\nStarting HRM training for {hrm_config.training.epochs} epochs...")

    try:
        model, best_f1 = run_training_with_tracking_hetero(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=hrm_config
        )

        print(f"\nHRM training completed! Best validation F1: {best_f1:.4f}")
        return model, best_f1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, 0.0
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

def main_benchmark_training():
    """Train benchmark comparison model."""
    set_seed(42)
    bm_config = Config(**BASELINE_CONFIG)

    train_loader, val_loader = get_benchmark_loaders(
        bm_config.data.train_json,
        batch_size=bm_config.training.batch_size,
        val_ratio=bm_config.training.val_ratio,
        seed=bm_config.data.seed,
        num_workers=bm_config.data.num_workers,
        pin_memory=bm_config.data.pin_memory,
        shuffle_train=bm_config.data.shuffle_train,
    )
    model = BenchmarkComparisonModel(
        input_dim=bm_config.benchmark.input_dim,
        hidden_dim=bm_config.benchmark.hidden_dim,
        layers=bm_config.benchmark.layers,
        p_drop=bm_config.benchmark.dropout
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"\nStarting Benchmark training for {bm_config.training.epochs} epochs...")
    try:
        model, best_f1 = benchmark_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=bm_config
        )

        print(f"\nBenchmark training completed! Best validation F1: {best_f1:.4f}")
        return model, best_f1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, 0.0
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


def run_hrm_baseline():
    """Run the HRM baseline experiment."""
    return main_hrm_training()

def run_heterogeneous_baseline():
    """Run the heterogeneous baseline experiment."""
    return main_heterogeneous_training()

def run_benchmark_training():
    """Run the benchmark training experiment."""
    return main_benchmark_training()
