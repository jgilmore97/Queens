import random
import torch
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import os

from model import GAT, HeteroGAT, HRM
from data_loader import get_homogeneous_loaders, get_queens_loaders
from train import run_training_with_tracking, run_training_with_tracking_hetero
from config import Config

MODELS_TO_TRAIN = ['gat', 'hetero_gat', 'hrm']
CHECKPOINT_BASE_DIR = 'checkpoints/comparison'
RESULTS_DIR = 'results'

DATA_CONFIG = {
    'state_training': 'data/StateTrainingSet.json',
    'state0_training': 'data/State0TrainingSet.json',
}

TRAINING_OVERRIDES = {
    'epochs': 18,
    'batch_size': 512,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'val_ratio': 0.10,
    'switch_epoch': 5,
    'mixed_ratio': 0.75,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
}

MODEL_CONFIG_OVERRIDES = {
    'input_dim': 14,
    'hidden_dim': 128,
    'layer_count': 6,
    'dropout': 0.2,
    'gat_heads': 2,
    'hgt_heads': 6,
    'n_cycles': 3,
    't_micro': 2,
    'use_input_injection': True,
}

SEED = 42
DISABLE_WANDB = True

def setup_directories():
    Path(CHECKPOINT_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for model_name in MODELS_TO_TRAIN:
        (Path(CHECKPOINT_BASE_DIR) / model_name).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created:")
    print(f"  - Checkpoints: {CHECKPOINT_BASE_DIR}/")
    print(f"  - Results: {RESULTS_DIR}/")

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (training will be slower)")
    return device

def create_config(model_name: str, model_type: str) -> Config:
    config = Config()
    config.model.model_type = model_type
    for key, value in MODEL_CONFIG_OVERRIDES.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
    for key, value in TRAINING_OVERRIDES.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
    config.data.train_json = DATA_CONFIG['state_training']
    config.training.state0_json_path = DATA_CONFIG['state0_training']
    config.experiment.experiment_name = f"comparison_{model_name}"
    config.experiment.checkpoint_dir = str(Path(CHECKPOINT_BASE_DIR) / model_name)
    config.experiment.tags = ['comparison', model_name]
    return config

def copy_best_checkpoint(model_name: str):
    source_dir = Path(CHECKPOINT_BASE_DIR) / model_name
    possible_names = ['best_model.pt', 'checkpoint_best.pt', 'model_best.pt']
    source_path = None
    for name in possible_names:
        candidate = source_dir / name
        if candidate.exists():
            source_path = candidate
            break
    if source_path is None:
        checkpoints = list(source_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            source_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    if source_path:
        dest_path = source_dir / 'best_model.pt'
        if source_path != dest_path:
            import shutil
            shutil.copy2(source_path, dest_path)
        print(f"✓ Checkpoint saved to {dest_path}")
        return dest_path
    else:
        print(f"⚠ No checkpoint found for {model_name}")
        return None

def plot_training_curves_from_checkpoints(model_names, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training Comparison', fontsize=16, fontweight='bold')
    colors = {'gat': '#1f77b4', 'hetero_gat': '#ff7f0e', 'hrm': '#2ca02c'}
    labels = {'gat': 'GAT (Homogeneous)', 'hetero_gat': 'HeteroGAT', 'hrm': 'HRM'}
    for model_name in model_names:
        checkpoint_path = Path(CHECKPOINT_BASE_DIR) / model_name / 'best_model.pt'
        if not checkpoint_path.exists():
            print(f"⚠ Checkpoint not found for {model_name}, skipping plot")
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            metrics = checkpoint.get('metrics', {})
            print(f"✓ Loaded metrics for {model_name}")
        except Exception as e:
            print(f"⚠ Could not load metrics for {model_name}: {e}")
    fig.text(0.5, 0.02,
             'Note: Training curves require W&B logging or modified checkpoint saving.\n'
             'Run compare_models.py for comprehensive evaluation.',
             ha='center', fontsize=10, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved placeholder plot to {save_path}")
    plt.close()

def train_gat_model(device):
    print("\n" + "="*70)
    print("TRAINING GAT (Homogeneous Graph Attention Network)")
    print("="*70)
    config = create_config('gat', 'GAT')
    model = GAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        heads=config.model.gat_heads
    )
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_loader, val_loader = get_homogeneous_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=config.training.val_ratio,
        seed=SEED,
        num_workers=0,
        pin_memory=True,
        shuffle_train=True,
    )
    print(f"✓ Train samples: {len(train_loader.dataset):,}")
    print(f"✓ Val samples: {len(val_loader.dataset):,}")
    print(f"\nTraining for {config.training.epochs} epochs...")
    if DISABLE_WANDB:
        os.environ['WANDB_MODE'] = 'disabled'
    model, best_f1 = run_training_with_tracking(
        model, train_loader, val_loader, config
    )
    print(f"\n✓ GAT Training Complete! Best F1: {best_f1:.4f}")
    copy_best_checkpoint('gat')
    return model

def train_hetero_gat_model(device):
    print("\n" + "="*70)
    print("TRAINING HeteroGAT (Heterogeneous Graph Attention Network)")
    print("="*70)
    config = create_config('hetero_gat', 'HeteroGAT')
    model = HeteroGAT(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        layer_count=config.model.layer_count,
        dropout=config.model.dropout,
        gat_heads=config.model.gat_heads,
        hgt_heads=config.model.hgt_heads,
        use_batch_norm=True
    )
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Edge types: line_constraint, region_constraint, diagonal_constraint")
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=config.training.val_ratio,
        seed=SEED,
        num_workers=0,
        pin_memory=True,
        shuffle_train=True,
    )
    print(f"✓ Train samples: {len(train_loader.dataset):,}")
    print(f"✓ Val samples: {len(val_loader.dataset):,}")
    print(f"\nTraining for {config.training.epochs} epochs...")
    if DISABLE_WANDB:
        os.environ['WANDB_MODE'] = 'disabled'
    model, best_f1 = run_training_with_tracking_hetero(
        model, train_loader, val_loader, config
    )
    print(f"\n✓ HeteroGAT Training Complete! Best F1: {best_f1:.4f}")
    copy_best_checkpoint('hetero_gat')
    return model

def train_hrm_model(device):
    print("\n" + "="*70)
    print("TRAINING HRM (Hierarchical Reasoning Model)")
    print("="*70)
    config = create_config('hrm', 'HRM')
    model = HRM(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        gat_heads=config.model.gat_heads,
        hgt_heads=config.model.hgt_heads,
        dropout=config.model.dropout,
        use_batch_norm=True,
        n_cycles=config.model.n_cycles,
        t_micro=config.model.t_micro,
        use_input_injection=config.model.use_input_injection,
    )
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ HRM Config: {config.model.n_cycles} cycles, {config.model.t_micro} micro-steps")
    print("✓ Edge types: line_constraint, region_constraint, diagonal_constraint")
    train_loader, val_loader = get_queens_loaders(
        config.data.train_json,
        batch_size=config.training.batch_size,
        val_ratio=config.training.val_ratio,
        seed=SEED,
        num_workers=0,
        pin_memory=True,
        shuffle_train=True,
    )
    print(f"✓ Train samples: {len(train_loader.dataset):,}")
    print(f"✓ Val samples: {len(val_loader.dataset):,}")
    print(f"\nTraining for {config.training.epochs} epochs...")
    if DISABLE_WANDB:
        os.environ['WANDB_MODE'] = 'disabled'
    model, best_f1 = run_training_with_tracking_hetero(
        model, train_loader, val_loader, config
    )
    print(f"\n✓ HRM Training Complete! Best F1: {best_f1:.4f}")
    copy_best_checkpoint('hrm')
    return model

def main():
    print("\n" + "="*70)
    print("QUEENS PUZZLE - MODEL COMPARISON TRAINING")
    print("="*70)
    print(f"Models to train: {', '.join([m.upper() for m in MODELS_TO_TRAIN])}")
    print(f"Epochs per model: {TRAINING_OVERRIDES['epochs']}")
    print(f"Dataset switch at epoch: {TRAINING_OVERRIDES['switch_epoch']}")
    if DISABLE_WANDB:
        print("W&B tracking: DISABLED")
    print("="*70 + "\n")
    setup_directories()
    device = get_device()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    training_results = {}
    start_time = time.time()
    for model_name in MODELS_TO_TRAIN:
        model_start_time = time.time()
        try:
            if model_name == 'gat':
                model = train_gat_model(device)
            elif model_name == 'hetero_gat':
                model = train_hetero_gat_model(device)
            elif model_name == 'hrm':
                model = train_hrm_model(device)
            else:
                print(f"⚠ Unknown model: {model_name}, skipping...")
                continue
            model_time = time.time() - model_start_time
            training_results[model_name] = {
                'training_time': model_time,
                'status': 'completed'
            }
            print(f"\n✓ {model_name.upper()} completed in {model_time/60:.1f} minutes\n")
        except Exception as e:
            print(f"\n⚠ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            training_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    total_time = time.time() - start_time
    plot_training_curves_from_checkpoints(
        [m for m in MODELS_TO_TRAIN if training_results.get(m, {}).get('status') == 'completed'],
        Path(RESULTS_DIR) / 'training_curves.png'
    )
    summary = {
        'training_config': TRAINING_OVERRIDES,
        'model_config': MODEL_CONFIG_OVERRIDES,
        'results': training_results,
        'total_training_time': total_time
    }
    summary_path = Path(RESULTS_DIR) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\nResults saved to:")
    print(f"  - {summary_path}")
    print(f"\nCheckpoints saved to:")
    for model_name in MODELS_TO_TRAIN:
        if training_results.get(model_name, {}).get('status') == 'completed':
            ckpt_path = Path(CHECKPOINT_BASE_DIR) / model_name / 'best_model.pt'
            if ckpt_path.exists():
                print(f"  - {ckpt_path}")
    print("\n" + "="*70)
    print("\nNext step: Run 'python compare_models.py' to evaluate all models")
    print("="*70)

if __name__ == "__main__":
    main()
