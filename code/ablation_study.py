import random
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from data_loader import HomogeneousQueensDataset
from train import train_model_for_ablation
from vectorized_eval import evaluate_solve_rate as evaluate_solve_rate_hetero, _batched_argmax
from bm_vectorized_eval import evaluate_solve_rate as evaluate_solve_rate_benchmark


PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_BASE_DIR = PROJECT_ROOT / 'checkpoints' / 'ablation'
RESULTS_DIR = PROJECT_ROOT / 'results'

DATA_PATHS = {
    'multistate_train': str(PROJECT_ROOT / 'data' / 'StateTrainingSet.json'),
    'state0_train': str(PROJECT_ROOT / 'data' / 'State0TrainingSet.json'),
    'test': str(PROJECT_ROOT / 'data' / 'FullSolveTestSet.json'),
}

SHARED_CONFIG = {
    'epochs': 12,
    'batch_size': 512,
    'learning_rate': 1.5e-3,
    'weight_decay': 0.000003,
    'val_ratio': 0.10,
    'seed': 42,
    'focal_alpha': 0.37,
    'focal_gamma': 2.2,
    'scheduler_type': 'cosine',
    'cosine_t_max': 12,
    'cosine_eta_min': 1e-6,
    'input_dim': 14,
    'hidden_dim': 128,
    'dropout': 0.12,
    'gat_heads': 4,
    'hgt_heads': 4,
    'hmod_heads': 4,
    'layer_count': 6,
    'layers': 6,
    'n_cycles': 3,
    't_micro': 2,
}

MODELS_TO_TRAIN = ['gat', 'hetero_gat', 'hrm_fullspatial', 'benchmark']

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_directories():
    CHECKPOINT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS_TO_TRAIN:
        (CHECKPOINT_BASE_DIR / model_name).mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class HomogeneousStep0Dataset(HomogeneousQueensDataset):
    def __init__(self, json_path: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__(json_path, split="all", val_ratio=val_ratio, seed=seed)
        self.records = [r for r in self.records if r.get('step', 0) == 0]


def _evaluate_batch_homogeneous(
    model: torch.nn.Module,
    batch,
    device: str,
    n: int,
    dataset: HomogeneousStep0Dataset,
    dataset_indices: List[int]
) -> Tuple[int, int, Dict[int, int], List[Dict]]:
    x = batch.x.clone()
    y = batch.y
    edge_index = batch.edge_index
    batch_indices = batch.batch
    num_graphs = batch.num_graphs

    still_correct = torch.ones(num_graphs, dtype=torch.bool, device=device)
    first_error_step = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
    placed = torch.zeros_like(y, dtype=torch.bool)

    for step in range(n):
        logits = model(x, edge_index)

        masked_logits = logits.clone()
        masked_logits[placed] = float('-inf')

        selected_nodes = _batched_argmax(masked_logits, batch_indices, num_graphs)

        selected_labels = y[selected_nodes]
        is_correct = (selected_labels == 1)

        just_failed = still_correct & ~is_correct
        first_error_step[just_failed] = step
        still_correct = still_correct & is_correct

        placed[selected_nodes] = True
        x[selected_nodes, -1] = 1.0

    solved_count = still_correct.sum().item()

    error_by_step = {}
    for step in range(n):
        count = (first_error_step == step).sum().item()
        if count > 0:
            error_by_step[step] = count

    failed_puzzles = []
    failed_mask = ~still_correct
    failed_indices = torch.where(failed_mask)[0].cpu().tolist()

    for local_idx in failed_indices:
        dataset_idx = dataset_indices[local_idx]
        record = dataset.records[dataset_idx]

        failed_puzzles.append({
            'source': record.get('source', f'puzzle_{dataset_idx}'),
            'board_size': len(record['region']),
            'first_error_step': first_error_step[local_idx].item()
        })

    return solved_count, num_graphs, error_by_step, failed_puzzles


def evaluate_solve_rate_homogeneous(
    model: torch.nn.Module,
    val_json_path: str,
    device: str,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    model.eval()

    dataset = HomogeneousStep0Dataset(val_json_path, val_ratio=val_ratio, seed=seed)

    size_groups = {}
    for idx in range(len(dataset)):
        record = dataset.records[idx]
        n = len(record['region'])
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(idx)

    total_puzzles = 0
    solved_puzzles = 0
    error_by_step = {}
    failed_puzzles = []

    with torch.no_grad():
        for n, indices in size_groups.items():
            for batch_start in range(0, len(indices), batch_size):
                batch_end = min(batch_start + batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]

                batch_data = [dataset.get(idx) for idx in batch_indices]
                batch = Batch.from_data_list(batch_data)
                batch = batch.to(device)

                batch_solved, batch_total, batch_errors, batch_failures = _evaluate_batch_homogeneous(
                    model, batch, device, n, dataset, batch_indices
                )

                solved_puzzles += batch_solved
                total_puzzles += batch_total
                failed_puzzles.extend(batch_failures)

                for step, count in batch_errors.items():
                    error_by_step[step] = error_by_step.get(step, 0) + count

    solve_rate = solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0

    return {
        'solve_rate': solve_rate,
        'total_puzzles': total_puzzles,
        'solved_puzzles': solved_puzzles,
        'error_by_step': error_by_step,
        'failed_puzzles': failed_puzzles
    }


def evaluate_model(model_name: str, model: torch.nn.Module, device) -> Dict:
    print(f"\nEvaluating {model_name.upper()} on test set...")

    test_path = DATA_PATHS['test']

    if model_name == 'gat':
        results = evaluate_solve_rate_homogeneous(model, test_path, device)
    elif model_name == 'benchmark':
        results = evaluate_solve_rate_benchmark(model, test_path, device)
    else:
        results = evaluate_solve_rate_hetero(model, test_path, device)

        failed_puzzles = results.get('failed_puzzles', [])
        if failed_puzzles and hasattr(failed_puzzles[0], 'source'):
            results['failed_puzzles'] = [
                {
                    'source': fp.source,
                    'board_size': fp.board_size,
                    'first_error_step': fp.first_error_step
                }
                for fp in failed_puzzles
            ]

    print(f"{model_name.upper()}: {results['solve_rate']:.1%} solve rate ({results['solved_puzzles']}/{results['total_puzzles']})")

    return results


def print_results_table(results: Dict[str, Dict]):
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Solve Rate':<12} {'Solved/Total':<15} {'Params':<12} {'Train Time':<12}")
    print("-"*80)

    for model_name in MODELS_TO_TRAIN:
        if model_name not in results:
            continue

        r = results[model_name]

        if 'error' in r:
            print(f"{model_name:<20} {'ERROR':<12} {'-':<15} {'-':<12} {'-':<12}")
            continue

        solve_rate = f"{r['solve_rate']:.1%}"
        solved_total = f"{r['solved_puzzles']}/{r['total_puzzles']}"
        params = f"{r['model_parameters']/1e6:.2f}M"
        train_time = f"{r['training_time']/60:.1f} min"

        print(f"{model_name:<20} {solve_rate:<12} {solved_total:<15} {params:<12} {train_time:<12}")

    print("="*80)


def print_failed_puzzles_summary(results: Dict[str, Dict]):
    print("\n" + "="*80)
    print("FAILED PUZZLES SUMMARY")
    print("="*80)

    for model_name in MODELS_TO_TRAIN:
        if model_name not in results:
            continue

        r = results[model_name]

        if 'error' in r:
            print(f"\n{model_name.upper()}: Training/evaluation failed")
            continue

        failed = r.get('failed_puzzles', [])

        print(f"\n{model_name.upper()} ({len(failed)} failures):")
        for fp in failed:
            source = fp['source'] if isinstance(fp, dict) else fp.source
            board_size = fp['board_size'] if isinstance(fp, dict) else fp.board_size
            first_error_step = fp['first_error_step'] if isinstance(fp, dict) else fp.first_error_step
            print(f"  - {source} ({board_size}x{board_size}, error at step {first_error_step})")
    print("="*80)


def save_results(results: Dict[str, Dict]):
    results_path = RESULTS_DIR / 'ablation_study.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    summary_path = RESULTS_DIR / 'ablation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("QUEENS PUZZLE - ABLATION STUDY RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Model':<20} {'Solve Rate':<12} {'Solved/Total':<15} {'Params':<12} {'Train Time':<12}\n")
        f.write("-"*80 + "\n")

        for model_name in MODELS_TO_TRAIN:
            if model_name not in results:
                continue

            r = results[model_name]

            if 'error' in r:
                f.write(f"{model_name:<20} {'ERROR':<12} {'-':<15} {'-':<12} {'-':<12}\n")
                f.write(f"  Error: {r['error']}\n")
                continue

            solve_rate = f"{r['solve_rate']:.1%}"
            solved_total = f"{r['solved_puzzles']}/{r['total_puzzles']}"
            params = f"{r['model_parameters']/1e6:.2f}M"
            train_time = f"{r['training_time']/60:.1f} min"

            f.write(f"{model_name:<20} {solve_rate:<12} {solved_total:<15} {params:<12} {train_time:<12}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("\nFAILED PUZZLES SUMMARY\n")
        f.write("="*80 + "\n")

        for model_name in MODELS_TO_TRAIN:
            if model_name not in results:
                continue

            r = results[model_name]
            failed = r.get('failed_puzzles', [])

            f.write(f"\n{model_name.upper()} ({len(failed)} failures):\n")
            for fp in failed:
                source = fp['source'] if isinstance(fp, dict) else fp.source
                board_size = fp['board_size'] if isinstance(fp, dict) else fp.board_size
                first_error_step = fp['first_error_step'] if isinstance(fp, dict) else fp.first_error_step
                f.write(f"  - {source} ({board_size}x{board_size}, error at step {first_error_step})\n")

    print(f"Summary saved to {summary_path}")


def main():
    print("\n" + "="*80)
    print("QUEENS PUZZLE - ABLATION STUDY")
    print("="*80)
    print(f"Models: {', '.join([m.upper() for m in MODELS_TO_TRAIN])}")
    print(f"Epochs: {SHARED_CONFIG['epochs']} | Batch Size: {SHARED_CONFIG['batch_size']} | LR: {SHARED_CONFIG['learning_rate']}")
    print(f"Test Set: {DATA_PATHS['test']}")
    print("="*80)

    setup_directories()
    device = get_device()
    set_seed(SHARED_CONFIG['seed'])

    results = {}

    for model_name in MODELS_TO_TRAIN:
        try:
            checkpoint_dir = str(CHECKPOINT_BASE_DIR / model_name)

            model, best_f1, training_time = train_model_for_ablation(
                model_type=model_name,
                multistate_json=DATA_PATHS['multistate_train'],
                state0_json=DATA_PATHS['state0_train'],
                test_json=DATA_PATHS['test'],
                checkpoint_dir=checkpoint_dir,
                config_overrides=SHARED_CONFIG,
                device=device
            )

            eval_results = evaluate_model(model_name, model, device)

            results[model_name] = {
                'model_parameters': count_parameters(model),
                'training_time': training_time,
                'best_val_f1': best_f1,
                **eval_results
            }

        except Exception as e:
            print(f"\nError with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    print_results_table(results)
    print_failed_puzzles_summary(results)
    save_results(results)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
