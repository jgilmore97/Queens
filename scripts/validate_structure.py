#!/usr/bin/env python3
"""
Validate that the restructured package imports and basic functionality work.
Run with: python scripts/validate_structure.py
"""

import sys
from pathlib import Path

# Add src to path for testing without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_dependencies():
    """Check if required dependencies are installed."""
    print("=" * 60)
    print("Checking dependencies...")
    print("=" * 60)

    missing = []

    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("[MISSING] torch")

    try:
        import torch_geometric
        print(f"[OK] torch_geometric {torch_geometric.__version__}")
    except ImportError:
        missing.append("torch-geometric")
        print("[MISSING] torch-geometric")

    try:
        import numpy
        print(f"[OK] numpy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
        print("[MISSING] numpy")

    try:
        import wandb
        print(f"[OK] wandb {wandb.__version__}")
    except ImportError:
        missing.append("wandb")
        print("[MISSING] wandb (optional)")

    return missing


def test_imports():
    """Test that all package imports work."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    errors = []

    # Config
    try:
        from queens_solver.config import Config, BASELINE_CONFIG
        print("[OK] queens_solver.config")
    except ImportError as e:
        errors.append(f"queens_solver.config: {e}")
        print(f"[FAIL] queens_solver.config: {e}")

    # Models
    try:
        from queens_solver.models.models import GAT, HeteroGAT, HRM, HRM_FullSpatial
        print("[OK] queens_solver.models.models")
    except ImportError as e:
        errors.append(f"queens_solver.models.models: {e}")
        print(f"[FAIL] queens_solver.models.models: {e}")

    try:
        from queens_solver.models.benchmark import BenchmarkHRM, BenchmarkSequential
        print("[OK] queens_solver.models.benchmark")
    except ImportError as e:
        errors.append(f"queens_solver.models.benchmark: {e}")
        print(f"[FAIL] queens_solver.models.benchmark: {e}")

    # Data
    try:
        from queens_solver.data.dataset import QueensDataset, get_queens_loaders
        print("[OK] queens_solver.data.dataset")
    except ImportError as e:
        errors.append(f"queens_solver.data.dataset: {e}")
        print(f"[FAIL] queens_solver.data.dataset: {e}")

    try:
        from queens_solver.data.preprocessing import extract, solve_queens
        print("[OK] queens_solver.data.preprocessing")
    except ImportError as e:
        errors.append(f"queens_solver.data.preprocessing: {e}")
        print(f"[FAIL] queens_solver.data.preprocessing: {e}")

    # Training
    try:
        from queens_solver.training.trainer import FocalLoss, run_training_with_tracking_hetero
        print("[OK] queens_solver.training.trainer")
    except ImportError as e:
        errors.append(f"queens_solver.training.trainer: {e}")
        print(f"[FAIL] queens_solver.training.trainer: {e}")

    try:
        from queens_solver.training.tracker import ExperimentTracker
        print("[OK] queens_solver.training.tracker")
    except ImportError as e:
        errors.append(f"queens_solver.training.tracker: {e}")
        print(f"[FAIL] queens_solver.training.tracker: {e}")

    try:
        from queens_solver.training.benchmark_trainer import benchmark_training
        print("[OK] queens_solver.training.benchmark_trainer")
    except ImportError as e:
        errors.append(f"queens_solver.training.benchmark_trainer: {e}")
        print(f"[FAIL] queens_solver.training.benchmark_trainer: {e}")

    # Evaluation
    try:
        from queens_solver.evaluation.solver import Solver
        print("[OK] queens_solver.evaluation.solver")
    except ImportError as e:
        errors.append(f"queens_solver.evaluation.solver: {e}")
        print(f"[FAIL] queens_solver.evaluation.solver: {e}")

    try:
        from queens_solver.evaluation.evaluator import evaluate_solve_rate
        print("[OK] queens_solver.evaluation.evaluator")
    except ImportError as e:
        errors.append(f"queens_solver.evaluation.evaluator: {e}")
        print(f"[FAIL] queens_solver.evaluation.evaluator: {e}")

    return errors


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing model instantiation...")
    print("=" * 60)

    errors = []

    from queens_solver.config import Config, BASELINE_CONFIG
    config = Config(**BASELINE_CONFIG)

    # HRM_FullSpatial (main model)
    try:
        from queens_solver.models.models import HRM_FullSpatial
        model = HRM_FullSpatial(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            gat_heads=config.model.gat_heads,
            hgt_heads=config.model.hgt_heads,
            hmod_heads=config.model.hmod_heads,
            dropout=config.model.dropout,
            n_cycles=config.model.n_cycles,
            t_micro=config.model.t_micro,
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] HRM_FullSpatial instantiated ({param_count:,} params)")
    except Exception as e:
        errors.append(f"HRM_FullSpatial: {e}")
        print(f"[FAIL] HRM_FullSpatial: {e}")

    # HeteroGAT
    try:
        from queens_solver.models.models import HeteroGAT
        model = HeteroGAT(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            layer_count=config.model.layer_count,
            dropout=config.model.dropout,
            gat_heads=config.model.gat_heads,
            hgt_heads=config.model.hgt_heads,
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] HeteroGAT instantiated ({param_count:,} params)")
    except Exception as e:
        errors.append(f"HeteroGAT: {e}")
        print(f"[FAIL] HeteroGAT: {e}")

    # GAT
    try:
        from queens_solver.models.models import GAT
        model = GAT(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            layer_count=config.model.layer_count,
            dropout=config.model.dropout,
            heads=config.model.gat_heads,
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] GAT instantiated ({param_count:,} params)")
    except Exception as e:
        errors.append(f"GAT: {e}")
        print(f"[FAIL] GAT: {e}")

    # BenchmarkHRM
    try:
        from queens_solver.models.benchmark import BenchmarkHRM
        model = BenchmarkHRM(
            input_dim=config.benchmark.input_dim,
            hidden_dim=config.benchmark.hidden_dim,
            p_drop=config.benchmark.dropout,
            n_heads=config.benchmark.n_heads,
            n_cycles=config.benchmark.n_cycles,
            t_micro=config.benchmark.microsteps,
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] BenchmarkHRM instantiated ({param_count:,} params)")
    except Exception as e:
        errors.append(f"BenchmarkHRM: {e}")
        print(f"[FAIL] BenchmarkHRM: {e}")

    return errors


def test_focal_loss():
    """Test FocalLoss instantiation and forward pass."""
    print("\n" + "=" * 60)
    print("Testing FocalLoss...")
    print("=" * 60)

    errors = []

    try:
        import torch
        from queens_solver.training.trainer import FocalLoss

        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Dummy forward pass
        logits = torch.randn(64)
        targets = torch.randint(0, 2, (64,)).float()
        loss = criterion(logits, targets)

        print(f"[OK] FocalLoss forward pass (loss={loss.item():.4f})")
    except Exception as e:
        errors.append(f"FocalLoss: {e}")
        print(f"[FAIL] FocalLoss: {e}")

    return errors


def test_scripts_importable():
    """Test that entry point scripts can be imported."""
    print("\n" + "=" * 60)
    print("Testing script imports...")
    print("=" * 60)

    errors = []
    scripts_dir = Path(__file__).parent

    # train.py
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", scripts_dir / "train.py")
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        print("[OK] scripts/train.py imports successfully")
    except Exception as e:
        errors.append(f"scripts/train.py: {e}")
        print(f"[FAIL] scripts/train.py: {e}")

    # ablation.py
    try:
        spec = importlib.util.spec_from_file_location("ablation", scripts_dir / "ablation.py")
        ablation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ablation_module)
        print("[OK] scripts/ablation.py imports successfully")
    except Exception as e:
        errors.append(f"scripts/ablation.py: {e}")
        print(f"[FAIL] scripts/ablation.py: {e}")

    return errors


def main():
    print("\n" + "=" * 60)
    print("QUEENS SOLVER - STRUCTURE VALIDATION")
    print("=" * 60)

    # Check dependencies first
    missing_deps = check_dependencies()

    core_missing = [d for d in missing_deps if d in ("torch", "torch-geometric", "numpy")]
    if core_missing:
        print("\n" + "=" * 60)
        print("CANNOT CONTINUE - Missing core dependencies:")
        print("=" * 60)
        print(f"\nInstall with: pip install {' '.join(core_missing)}")
        print("\nOr install the full package: pip install -e .")
        print("\n[SKIP] Skipping remaining tests due to missing dependencies.")
        print("\nTo validate structure without full deps, check that files exist:")

        # At least verify file structure exists
        print("\n" + "=" * 60)
        print("Verifying file structure...")
        print("=" * 60)

        base = Path(__file__).parent.parent
        expected_files = [
            "src/queens_solver/__init__.py",
            "src/queens_solver/config.py",
            "src/queens_solver/models/models.py",
            "src/queens_solver/models/benchmark.py",
            "src/queens_solver/data/dataset.py",
            "src/queens_solver/data/preprocessing.py",
            "src/queens_solver/training/trainer.py",
            "src/queens_solver/training/tracker.py",
            "src/queens_solver/evaluation/solver.py",
            "src/queens_solver/evaluation/evaluator.py",
            "scripts/train.py",
            "scripts/ablation.py",
            "app/gradio_app.py",
            "pyproject.toml",
            "Makefile",
        ]

        all_exist = True
        for f in expected_files:
            path = base / f
            if path.exists():
                print(f"[OK] {f}")
            else:
                print(f"[MISSING] {f}")
                all_exist = False

        if all_exist:
            print("\nFile structure is correct!")
            print("Install dependencies to run full validation.")
            return 0
        else:
            print("\nSome files are missing!")
            return 1

    all_errors = []

    all_errors.extend(test_imports())
    all_errors.extend(test_model_instantiation())
    all_errors.extend(test_focal_loss())
    all_errors.extend(test_scripts_importable())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\n{len(all_errors)} error(s) found:\n")
        for err in all_errors:
            print(f"  - {err}")
        print("\nPlease fix the above issues.")
        return 1
    else:
        print("\nAll tests passed! Structure is valid.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
