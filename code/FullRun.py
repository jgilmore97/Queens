import os
import torch
from pathlib import Path

from train import run_training_with_tracking, FocalLoss, run_training_with_tracking_hetero, train_epoch_hetero, evaluate_epoch_hetero, calculate_top1_metrics_hetero, run_training_with_tracking_hrm
from model import GAT, HeteroGAT
from data_loader import get_queens_loaders, QueensDataset
from config import Config, BASELINE_CONFIG, HYPEROPT_CONFIG
from experiment_tracker_fixed import ExperimentTracker, create_wandb_sweep, EXAMPLE_SWEEP_CONFIG

import wandb
from torch_geometric.loader import DataLoader
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

def main_heterogeneous_training():
    """Main training function with heterogeneous GAT and experiment tracking."""
    
    set_seed(42)
    
    hetero_config = Config(**BASELINE_CONFIG)
    
    print("=== Queens Puzzle ML Training - HETEROGENEOUS ===")
    print(f"Device: {hetero_config.system.device}")
    print(f"Experiment: {hetero_config.experiment.experiment_name}")
    print("🔗 Using heterogeneous edges with constraint-specific attention")
    
    # Load data (train/val only)
    print("\n Loading datasets...")
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
    
    print(f"\n Creating HETEROGENEOUS {hetero_config.model.model_type} model...")
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
    
    print(f"\n Starting HETEROGENEOUS training for {hetero_config.training.epochs} epochs...")
    
    try:
        model, best_f1 = run_training_with_tracking_hetero(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=hetero_config
        )
        
        print(f"\n HETEROGENEOUS training completed! Best validation F1: {best_f1:.4f}")
        print("🎯 Each constraint type learned specialized attention patterns")
        print("Model checkpoints saved for future test evaluation")
        
        return model, best_f1
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return None, 0.0
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        raise

def main_hrm_training():
    """Main training function with HRM model."""
    
    set_seed(42)
    
    hrm_config = Config(**BASELINE_CONFIG)
 
    print("=== Queens Puzzle ML Training - HRM ===")
    print(f"Device: {hrm_config.system.device}")
    print(f"Experiment: {hrm_config.experiment.experiment_name}")
    print("Using Hierarchical Reasoning Model (HRM)")
    
    # Load data
    print("\nLoading datasets...")
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
    
    # Create HRM model
    from model import HRM
    
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
        z_init=hrm_config.model.z_init,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\nStarting HRM training for {hrm_config.training.epochs} epochs...")
    
    try:
        model, best_f1 = run_training_with_tracking_hrm(
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

def run_hrm_baseline():
    """Run the HRM baseline experiment."""
    return main_hrm_training()

def compare_homogeneous_vs_heterogeneous():
    """Run both homogeneous and heterogeneous models for comparison."""
    
    print("🔬 COMPARISON EXPERIMENT: Homogeneous vs Heterogeneous GAT")
    print("=" * 60)
    
    set_seed(42)
    
    config = Config(**BASELINE_CONFIG)
    config.data.train_json = "10k_training_set_with_states.json"
    config.training.epochs = 20
    config.training.batch_size = 512
    
    # Load data once
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=config.training.val_ratio,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
        shuffle_train=config.data.shuffle_train,
    )
    
    results = {}
    
    # 1. HOMOGENEOUS MODEL
    print("\n🏠 Training HOMOGENEOUS GAT...")
    homo_config = config
    homo_config.experiment.experiment_name = "comparison_homo_gat"
    homo_config.experiment.tags = ["comparison", "homogeneous", "gat"]
    
    homo_model = GAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        heads=config.model.heads
    )
    
    try:
        homo_model, homo_f1 = run_training_with_tracking(
            model=homo_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=homo_config
        )
        results['homogeneous'] = homo_f1
        print(f"✅ Homogeneous F1: {homo_f1:.4f}")
    except Exception as e:
        print(f"❌ Homogeneous training failed: {e}")
        results['homogeneous'] = 0.0
    
    # 2. HETEROGENEOUS MODEL  
    print("\n🔗 Training HETEROGENEOUS GAT...")
    hetero_config = config
    hetero_config.experiment.experiment_name = "comparison_hetero_gat"
    hetero_config.experiment.tags = ["comparison", "heterogeneous", "gat"]
    
    hetero_model = HeteroGAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        gat_heads=hetero_config.model.gat_heads,
        hgt_heads=hetero_config.model.hgt_heads 
    )
    
    try:
        hetero_model, hetero_f1 = run_training_with_tracking_hetero(
            model=hetero_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=hetero_config
        )
        results['heterogeneous'] = hetero_f1
        print(f"✅ Heterogeneous F1: {hetero_f1:.4f}")
    except Exception as e:
        print(f"❌ Heterogeneous training failed: {e}")
        results['heterogeneous'] = 0.0
    
    # RESULTS
    print("\n" + "=" * 60)
    print("🏆 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Homogeneous GAT:    {results['homogeneous']:.4f}")
    print(f"Heterogeneous GAT:  {results['heterogeneous']:.4f}")
    
    if results['heterogeneous'] > results['homogeneous']:
        improvement = results['heterogeneous'] - results['homogeneous']
        print(f"🎯 Heterogeneous WINS by {improvement:.4f} F1 points!")
        print("✨ Constraint-specific attention helps!")
    elif results['homogeneous'] > results['heterogeneous']:
        difference = results['homogeneous'] - results['heterogeneous']
        print(f"🤔 Homogeneous wins by {difference:.4f} F1 points")
        print("💭 Might need more tuning or the added complexity isn't beneficial")
    else:
        print("🤝 It's a tie! Both approaches work similarly well")
    
    return results

def quick_hetero_test():
    """Quick test run for debugging heterogeneous model."""
    print("🧪 Running quick heterogeneous test...")
    
    config = Config()
    config.training.epochs = 2
    config.training.batch_size = 128
    config.experiment.experiment_name = "quick_hetero_test"
    config.experiment.tags = ["test", "heterogeneous", "debug"]
    
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=0.2,
        num_workers=2,
    )
    
    config.model.hidden_dim = 128
    config.model.layer_count = 2
    
    model = HeteroGAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        gat_heads=config.model.gat_heads,
        hgt_heads=config.model.hgt_heads 
    )
    
    try:
        model, best_f1 = run_training_with_tracking_hetero(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        print(f"🎯 Quick heterogeneous test completed! Best validation F1: {best_f1:.4f}")
        return True
    except Exception as e:
        print(f"❌ Quick heterogeneous test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_hetero_data():
    """Debug heterogeneous data loading."""
    print("🔍 Debugging heterogeneous data format...")
    
    config = Config()
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=2,  # Small batch for debugging
        val_ratio=0.2,
        num_workers=0,  # No multiprocessing for easier debugging
    )
    
    print("Loading one batch...")
    batch = next(iter(train_loader))
    
    print(f"Batch type: {type(batch)}")
    print(f"Batch attributes: {dir(batch)}")
    
    if hasattr(batch, '__getitem__'):
        print("\nTrying to access heterogeneous structure...")
        try:
            print(f"Cell features shape: {batch['cell'].x.shape}")
            print(f"Cell labels shape: {batch['cell'].y.shape}")
            
            edge_types = []
            for edge_type in [('cell', 'line_constraint', 'cell'), 
                            ('cell', 'region_constraint', 'cell'),
                            ('cell', 'diagonal_constraint', 'cell')]:
                if edge_type in batch.edge_types:
                    edge_shape = batch[edge_type].edge_index.shape
                    print(f"Edge {edge_type}: {edge_shape}")
                    edge_types.append(edge_type)
            
            print(f"Available edge types: {edge_types}")
            
        except Exception as e:
            print(f"Error accessing batch structure: {e}")
            import traceback
            traceback.print_exc()
    
    return batch

# Helper functions for easy execution
def run_heterogeneous_baseline():
    """Run the heterogeneous baseline experiment - call this in a Colab cell."""
    return main_heterogeneous_training()

def run_model_comparison():
    """Run comparison between homogeneous and heterogeneous - call this in a Colab cell."""
    return compare_homogeneous_vs_heterogeneous()

def run_quick_hetero_debug():
    """Run quick heterogeneous debug - call this in a Colab cell."""
    return quick_hetero_test()

def debug_hetero_batch():
    """Debug heterogeneous data loading - call this in a Colab cell."""
    return debug_hetero_data()