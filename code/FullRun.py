import os
import torch
from pathlib import Path

# Import all necessary modules
from train import run_training_with_tracking, FocalLoss
from model import GAT
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

def main_training():
    """Main training function with experiment tracking."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create configuration
    config = Config(**BASELINE_CONFIG)
    
    # Override specific settings if needed
    config.data.train_json = "10k_training_set_with_states.json"
    config.data.test_json = "test_set_with_states.json"
    config.training.epochs = 30
    config.training.batch_size = 512
    
    # Set experiment details
    config.experiment.experiment_name = "baseline_gat_v1"
    config.experiment.tags = ["baseline", "gat", "focal_loss"]
    config.experiment.notes = "Baseline GAT model with improved tracking and focal loss"
    
    print("=== Queens Puzzle ML Training ===")
    print(f"Device: {config.system.device}")
    print(f"Experiment: {config.experiment.experiment_name}")
    
    # Load data (train/val only - no test set loading during experiments)
    print("\n📊 Loading datasets...")
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=config.training.val_ratio,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle_train=config.data.shuffle_train,
    )
    
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print("Test set reserved for final evaluation only")
    
    # Create model
    print(f"\n🧠 Creating {config.model.model_type} model...")
    model = GAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        heads=config.model.heads
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Run training with tracking
    print(f"\n🚀 Starting training for {config.training.epochs} epochs...")
    
    try:
        model, best_f1 = run_training_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        print(f"\n✅ Training completed! Best validation F1: {best_f1:.4f}")
        print("📝 Model checkpoints saved for future test evaluation")
        
        return model, best_f1
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return None, 0.0
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

def evaluate_test_set(model, test_loader, config):
    """
    RESERVED FOR FINAL MODEL EVALUATION ONLY!
    
    This function should only be called with your final, best model
    after all experimentation is complete. Using the test set during
    development can lead to overfitting to the test data.
    """
    print("⚠️ WARNING: Evaluating on TEST SET")
    print("   This should only be done with your final model!")
    
    device = config.system.device
    model.eval()
    
    criterion = FocalLoss(
        alpha=config.training.focal_alpha,
        gamma=config.training.focal_gamma
    )
    
    total_loss, total_nodes = 0.0, 0
    correct, TP, FP, FN, TN = 0, 0, 0, 0, 0
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y.float())
            
            preds = (logits > 0).long()
            
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
            correct += (preds == batch.y).sum().item()
            
            TP += ((preds == 1) & (batch.y == 1)).sum().item()
            FP += ((preds == 1) & (batch.y == 0)).sum().item()
            FN += ((preds == 0) & (batch.y == 1)).sum().item()
            TN += ((preds == 0) & (batch.y == 0)).sum().item()
    
    # Calculate metrics
    eps = 1e-9
    test_loss = total_loss / total_nodes
    test_acc = correct / total_nodes
    test_prec = TP / (TP + FP + eps)
    test_rec = TP / (TP + FN + eps)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + eps)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1
    }

def final_test_evaluation(model_checkpoint_path, config):
    """
    Load best model and evaluate on test set.
    Only use this function when you're done with all experiments!
    """
    print("🧪 FINAL TEST EVALUATION")
    print("Loading test dataset...")
    
    test_dataset = QueensDataset(
        config.data.test_json,
        split="all",
        val_ratio=0.0,
        seed=config.data.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    print(f"Test samples: {len(test_dataset):,}")
    
    # Load model
    print(f"Loading model from {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=config.system.device)
    
    model = GAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        heads=config.model.heads
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    test_metrics = evaluate_test_set(model, test_loader, config)
    
    print("\n🎯 FINAL TEST RESULTS:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    return test_metrics

def run_hyperparameter_sweep():
    """Run hyperparameter optimization sweep."""
    print("🔍 Setting up hyperparameter sweep...")
    
    # Create sweep
    sweep_id = create_wandb_sweep(
        config=EXAMPLE_SWEEP_CONFIG,
        project_name="queens-puzzle-ml"
    )
    
    def sweep_train():
        """Training function for sweep."""
        with wandb.init() as run:
            # Get hyperparameters from wandb
            sweep_config = wandb.config
            
            # Create config with sweep parameters
            config = Config(**HYPEROPT_CONFIG)
            
            # Update with sweep parameters
            if hasattr(sweep_config, 'learning_rate'):
                config.training.learning_rate = sweep_config.learning_rate
            if hasattr(sweep_config, 'hidden_dim'):
                config.model.hidden_dim = sweep_config.hidden_dim
            if hasattr(sweep_config, 'dropout'):
                config.model.dropout = sweep_config.dropout
            if hasattr(sweep_config, 'focal_gamma'):
                config.training.focal_gamma = sweep_config.focal_gamma
            
            # Set experiment name
            config.experiment.experiment_name = f"sweep_{run.id}"
            
            # Load data (train/val only)
            train_loader, val_loader = get_queens_loaders(
                config.data.train_json,
                batch_size=config.training.batch_size,
                val_ratio=config.training.val_ratio,
                seed=config.data.seed,
                num_workers=2,  # Reduce for sweep
            )
            
            # Create model
            model = GAT(
                input_dim=config.model.input_dim,
                hidden_dim=config.model.hidden_dim,
                layer_count=config.model.layer_count,
                dropout=config.model.dropout,
                heads=config.model.heads
            )
            
            # Run training
            try:
                _, best_f1 = run_training_with_tracking(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config
                )
                return best_f1
            except Exception as e:
                print(f"Sweep run failed: {e}")
                return 0.0
    
    print(f"Sweep ID: {sweep_id}")
    print("To run sweep agents, execute this in a cell:")
    print(f"!wandb agent {sweep_id}")
    
    return sweep_id, sweep_train

def quick_test():
    """Quick test run for debugging."""
    print("🧪 Running quick test...")
    
    config = Config()
    config.training.epochs = 2
    config.training.batch_size = 128
    config.experiment.experiment_name = "quick_test"
    config.experiment.tags = ["test", "debug"]
    
    # Load small subset of data (train/val only)
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=0.2,
        num_workers=2,
    )
    
    # Small model
    config.model.hidden_dim = 128
    config.model.layer_count = 2
    
    model = GAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        heads=config.model.heads
    )
    
    try:
        model, best_f1 = run_training_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        print(f"✅ Quick test completed! Best validation F1: {best_f1:.4f}")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

# Colab-friendly functions - call these directly in notebook cells
def run_baseline_experiment():
    """Run the baseline experiment - call this in a Colab cell."""
    return main_training()

def run_quick_debug():
    """Run a quick debug session - call this in a Colab cell."""
    return quick_test()

def setup_hyperparameter_sweep():
    """Set up hyperparameter sweep - call this in a Colab cell."""
    return run_hyperparameter_sweep()

def load_and_test_final_model(checkpoint_path="checkpoints/best_model.pt"):
    """
    ONLY use this when you're completely done with experiments!
    This loads your best model and evaluates it on the test set.
    """
    config = Config(**BASELINE_CONFIG)
    return final_test_evaluation(checkpoint_path, config)

# Helper function for Colab users
def print_usage_guide():
    """Print usage instructions for Colab."""
    print("📖 COLAB USAGE GUIDE")
    print("=" * 50)
    print()
    print("🔥 Quick Start:")
    print("   model, best_f1 = run_baseline_experiment()")
    print()
    print("🧪 Quick Debug (2 epochs):")
    print("   success = run_quick_debug()")
    print()
    print("🔍 Hyperparameter Sweep:")
    print("   sweep_id, sweep_fn = setup_hyperparameter_sweep()")
    print("   # Then run: !wandb agent {sweep_id}")
    print()
    print("🎯 Final Test (ONLY when done with ALL experiments):")
    print("   test_results = load_and_test_final_model()")
    print()
    print("📊 Customize experiment:")
    print("   config = Config(**BASELINE_CONFIG)")
    print("   config.training.epochs = 50")
    print("   config.experiment.experiment_name = 'my_experiment'")
    print("   # Then use config in run_training_with_tracking()")
    print()
    print("💡 Remember: Test set is reserved for final evaluation only!")

# Display the guide when imported
print_usage_guide()