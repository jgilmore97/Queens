import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from queens_solver.evaluation.solver import Solver, _SingleBatch
from queens_solver.models.models import HRM, HeteroGAT, GAT

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    'benchmark_sequential': {'subdir': 'benchmark_sequential', 'display_name': 'Benchmark-Seq', 'is_hrm': False},
    'benchmark_hrm': {'subdir': 'benchmark_hrm', 'display_name': 'Benchmark-HRM', 'is_hrm': False},
    'gat': {'subdir': 'gat', 'display_name': 'GAT', 'is_hrm': False},
    'hetero_gat': {'subdir': 'hetero_gat', 'display_name': 'HeteroGAT', 'is_hrm': False},
    'hrm_fullspatial': {'subdir': 'hrm_fullspatial', 'display_name': 'HRM', 'is_hrm': True},
}


class FailureAnalyzer(Solver):
    """
    Extended Solver class for failure analysis and model comparison.

    Inherits all solving capabilities from Solver and adds methods for:
    - Detailed diagnostic solving with step-by-step tracking
    - Analyzing failure patterns on datasets
    - Visualizing failure statistics
    - Comparing multiple models on the same puzzles
    - Attention weight capture and visualization
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__(model_path=model_path, device=device)
        self.stats = {
            'total': 0,
            'correct': 0,
            'all_sizes': [],
            'failed_sizes': [],
            'first_mistake_steps': [],
            'first_mistake_logits': []
        }

    def reset_stats(self):
        """Reset accumulated statistics."""
        self.stats = {
            'total': 0,
            'correct': 0,
            'all_sizes': [],
            'failed_sizes': [],
            'first_mistake_steps': [],
            'first_mistake_logits': []
        }

    def place_queen_with_attention(self, region_board: np.ndarray, partial_board: np.ndarray,
                                    edge_index_dict: Dict[str, torch.Tensor],
                                    capture_activations: bool = False,
                                    capture_attention: bool = False,
                                    activation_metric: str = 'l2_norm') -> Tuple:
        """Place a queen with optional attention weight capture."""
        n = region_board.shape[0]
        activation_dict = {}

        with torch.no_grad():
            if getattr(self, 'is_benchmark', False):
                node_features = self._build_node_features(region_board, partial_board, pad_for_benchmark=True)
                node_features = node_features.to(self.device)
                x_batched = node_features.unsqueeze(0)
                logits = self.model(x_batched)
                logits = logits.squeeze(0).squeeze(-1)
                logits_padded = logits.cpu().numpy().reshape(self.max_regions, self.max_regions)
                logits_np = logits_padded[:n, :n]

            elif isinstance(self.model, HRM):
                node_features = self._build_node_features(region_board, partial_board)
                node_features = node_features.to(self.device)
                edge_index_dict_formatted = {
                    ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                    ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                    ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
                }
                batch = _SingleBatch(node_features, edge_index_dict_formatted, self.device)
                if capture_activations or capture_attention:
                    logits, intermediates = self.model(
                        batch,
                        return_intermediates=capture_activations,
                        return_attention=capture_attention
                    )
                    if capture_activations:
                        activation_dict.update(self._process_intermediates(intermediates, n, activation_metric))
                    if capture_attention and 'H_attentions' in intermediates:
                        activation_dict['H_attentions'] = intermediates['H_attentions']
                        activation_dict['board_size'] = intermediates['board_size']
                else:
                    logits = self.model(batch)
                logits_np = logits.cpu().numpy().reshape(n, n)

            elif isinstance(self.model, HeteroGAT):
                node_features = self._build_node_features(region_board, partial_board)
                node_features = node_features.to(self.device)
                edge_index_dict_formatted = {
                    ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                    ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                    ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
                }
                batch = _SingleBatch(node_features, edge_index_dict_formatted, self.device)
                if capture_attention:
                    logits, attention_dict = self.model.forward(batch, return_attention=True)
                    activation_dict['gat_attention'] = attention_dict
                else:
                    logits = self.model(batch)
                logits_np = logits.cpu().numpy().reshape(n, n)

            elif isinstance(self.model, GAT):
                node_features = self._build_node_features(region_board, partial_board)
                node_features = node_features.to(self.device)
                edge_indices = [
                    edge_index_dict['line_constraint'],
                    edge_index_dict['region_constraint'],
                    edge_index_dict['diagonal_constraint'],
                ]
                combined_edge_index = torch.cat(edge_indices, dim=1)
                if capture_attention:
                    logits, attention_weights = self.model(node_features, combined_edge_index, return_attention=True)
                    activation_dict['gat_attention'] = attention_weights
                else:
                    logits = self.model(node_features, combined_edge_index)
                logits_np = logits.cpu().numpy().reshape(n, n)

            else:
                raise ValueError(f"Unknown model type: {type(self.model)}")

        flat_logits = logits_np.flatten()
        top_idx = np.argmax(flat_logits)
        top_logit = flat_logits[top_idx]
        top_row, top_col = top_idx // n, top_idx % n

        activation_dict['placement'] = (top_row, top_col)
        activation_dict['logit'] = float(top_logit)
        activation_dict['logits'] = logits_np

        return top_row, top_col, top_logit, activation_dict, logits_np

    def solve_puzzle_diagnostic(self, puzzle: dict, ground_truth: np.ndarray = None,
                                 top_k: int = 10, verbose: bool = True,
                                 capture_attention: bool = False,
                                 capture_activations: bool = False,
                                 batch_placement: bool = False,
                                 confidence_threshold: float = 4.0) -> dict:
        """Solve puzzle with detailed diagnostic information for failure analysis.

        Args:
            puzzle: Dict with 'region' key containing the region board.
            ground_truth: Ground truth solution. If None, computed via backtracking.
            top_k: Number of top predictions to track per pass.
            verbose: If True, print step-by-step diagnostic output.
            capture_attention: If True, capture attention weights for visualization.
            capture_activations: If True, capture intermediate activations (HRM only).
            batch_placement: If True, place multiple queens per pass when confident.
            confidence_threshold: Logit threshold for batch placement.

        Returns:
            Dict with solution, placement_order, correct, passes, forward_pass_count, activations
        """
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        queen_board = np.zeros((n, n), dtype=int)
        edge_index_dict = self._build_edge_index(region_board)

        if ground_truth is None:
            ground_truth = self.solve_with_vanilla_backtracking(puzzle)
        correct_positions = set(zip(*np.where(ground_truth == 1)))

        placement_order = []
        pass_diagnostics = []
        activations = []
        queens_placed = 0
        forward_pass_count = 0

        while queens_placed < n:
            forward_pass_count += 1

            _, _, _, act_dict, logits_np = self.place_queen_with_attention(
                region_board, queen_board, edge_index_dict,
                capture_activations=capture_activations,
                capture_attention=capture_attention
            )

            if capture_attention or capture_activations:
                activations.append(act_dict)

            remaining_correct = correct_positions - set(placement_order)
            flat_logits = logits_np.flatten()
            top_k_indices = np.argsort(flat_logits)[::-1][:top_k]

            top_k_predictions = []
            first_correct_rank = None
            for rank, idx in enumerate(top_k_indices, 1):
                pred_row, pred_col = idx // n, idx % n
                pred_logit = flat_logits[idx]
                is_correct = (pred_row, pred_col) in remaining_correct
                top_k_predictions.append({
                    'rank': rank,
                    'position': (pred_row, pred_col),
                    'logit': float(pred_logit),
                    'is_correct': is_correct
                })
                if is_correct and first_correct_rank is None:
                    first_correct_rank = rank

            pass_info = {
                'pass': forward_pass_count,
                'queens_placed_before': queens_placed,
                'top_k_predictions': top_k_predictions,
                'first_correct_rank': first_correct_rank,
                'remaining_correct': sorted(remaining_correct),
            }

            if batch_placement:
                rows, cols, logits = self._get_high_confidence_placements(
                    logits_np, queen_board, confidence_threshold
                )

                remaining = n - queens_placed
                if len(rows) > remaining:
                    rows, cols, logits = rows[:remaining], cols[:remaining], logits[:remaining]

                queen_board[rows, cols] = 1
                placed_positions = list(zip(rows.tolist(), cols.tolist()))
                placement_order.extend(placed_positions)
                queens_placed += len(rows)

                pass_info['queens_placed'] = placed_positions
                pass_info['logits_placed'] = logits.tolist()
                pass_info['above_threshold_count'] = int(np.sum(flat_logits > confidence_threshold))
            else:
                valid_mask = (queen_board == 0)
                masked_logits = np.where(valid_mask, logits_np, -np.inf)
                best_idx = np.argmax(masked_logits)
                row, col = best_idx // n, best_idx % n
                top_logit = masked_logits[row, col]

                pass_info['queens_placed'] = [(row, col)]
                pass_info['logits_placed'] = [float(top_logit)]
                pass_info['above_threshold_count'] = 0

                placement_order.append((row, col))
                queen_board[row, col] = 1
                queens_placed += 1

            pass_diagnostics.append(pass_info)

        is_correct = np.array_equal(queen_board, ground_truth)

        if verbose:
            for pass_info in pass_diagnostics:
                self._print_pass_diagnostic(pass_info, batch_placement)

            print(f"\n{'='*50}")
            print(f"FINAL RESULT: {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"Forward passes: {forward_pass_count}")
            print(f"{'='*50}")

        return {
            'solution': queen_board,
            'placement_order': placement_order,
            'correct': is_correct,
            'forward_pass_count': forward_pass_count,
            'passes': pass_diagnostics,
            'activations': activations if (capture_attention or capture_activations) else None
        }

    def _print_pass_diagnostic(self, pass_info: dict, batch_placement: bool = False) -> None:
        """Print formatted diagnostic output for a single forward pass."""
        print(f"\n=== PASS {pass_info['pass']} ===")
        print(f"Queens placed before this pass: {pass_info['queens_placed_before']}")
        print(f"Remaining correct positions: {pass_info['remaining_correct']}")
        print(f"Top {len(pass_info['top_k_predictions'])} predictions:")

        for pred in pass_info['top_k_predictions']:
            pos = pred['position']
            logit = pred['logit']
            marker = 'Y' if pred['is_correct'] else 'N'
            print(f"  Rank {pred['rank']:2d}: ({pos[0]},{pos[1]}) logit={logit:6.3f} {marker}")

        if pass_info['first_correct_rank'] is not None:
            print(f"First correct position appears at rank: {pass_info['first_correct_rank']}")
        else:
            print(f"First correct position appears at rank: >top-k")

        queens_placed = pass_info['queens_placed']
        logits_placed = pass_info['logits_placed']
        if batch_placement:
            print(f"Above threshold count: {pass_info['above_threshold_count']}")
        print(f"Placed {len(queens_placed)} queen(s) this pass:")
        for (r, c), logit in zip(queens_placed, logits_placed):
            print(f"  ({r},{c}) logit={logit:.3f}")

    def analyze_puzzle(self, puzzle: dict, ground_truth: Optional[np.ndarray] = None,
                       batch_placement: bool = True,
                       confidence_threshold: float = 4.0) -> dict:
        """
        Analyze a single puzzle and update internal statistics.

        Args:
            puzzle: Dict with 'region' key containing the region board.
            ground_truth: Optional pre-computed ground truth. If None, computed via backtracking.
            batch_placement: If True, place multiple queens per pass when confident.
            confidence_threshold: Logit threshold for batch placement.

        Returns:
            Dict with analysis results including correctness and failure details.
        """
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]

        if ground_truth is None:
            ground_truth = self.solve_with_vanilla_backtracking(puzzle)
        correct_positions = set(zip(*np.where(ground_truth == 1)))

        self.stats['total'] += 1
        self.stats['all_sizes'].append(n)

        result = self.solve_puzzle_diagnostic(
            puzzle,
            ground_truth=ground_truth,
            verbose=False,
            batch_placement=batch_placement,
            confidence_threshold=confidence_threshold
        )

        analysis = {
            'correct': result['correct'],
            'puzzle_size': n,
            'first_mistake_step': None,
            'first_mistake_logit': None
        }

        if result['correct']:
            self.stats['correct'] += 1
        else:
            self.stats['failed_sizes'].append(n)

            placed_queens = []
            queen_count = 0
            found_mistake = False
            for pass_info in result['passes']:
                placements = pass_info.get('queens_placed', [])
                logits_placed = pass_info.get('logits_placed', [])

                for i, placement in enumerate(placements):
                    queen_count += 1
                    remaining_correct = correct_positions - set(placed_queens)
                    is_correct = placement in remaining_correct

                    if not is_correct:
                        mistake_logit = logits_placed[i] if i < len(logits_placed) else 0.0

                        self.stats['first_mistake_steps'].append(queen_count)
                        self.stats['first_mistake_logits'].append(mistake_logit)

                        analysis['first_mistake_step'] = queen_count
                        analysis['first_mistake_logit'] = mistake_logit
                        found_mistake = True
                        break

                    placed_queens.append(placement)

                if found_mistake:
                    break

        return analysis

    def get_stats(self) -> dict:
        """Return current accumulated statistics."""
        return self.stats.copy()

    def get_accuracy(self) -> float:
        """Return accuracy as a percentage."""
        if self.stats['total'] == 0:
            return 0.0
        return self.stats['correct'] / self.stats['total'] * 100

    def print_summary(self):
        """Print a summary of accumulated statistics."""
        s = self.stats
        accuracy = self.get_accuracy()
        print(f"Accuracy: {s['correct']}/{s['total']} ({accuracy:.1f}%)")

        if s['first_mistake_logits']:
            logits = s['first_mistake_logits']
            print(f"First mistake logit - Median: {np.median(logits):.3f}, "
                  f"IQR: [{np.percentile(logits, 25):.3f}, {np.percentile(logits, 75):.3f}]")

        if s['first_mistake_steps']:
            steps = s['first_mistake_steps']
            print(f"First mistake step - Median: {np.median(steps):.1f}, "
                  f"IQR: [{np.percentile(steps, 25):.1f}, {np.percentile(steps, 75):.1f}]")


def evaluate_models_on_dataset(
    dataset_path: str,
    models_dir: str = 'checkpoints/app_models',
    output_dir: Optional[str] = None,
    device: str = 'cuda'
) -> Dict[str, dict]:
    """
    Evaluate multiple models on a dataset and collect failure statistics.

    Args:
        dataset_path: Path to JSON file containing puzzle records.
        models_dir: Path to directory containing model subdirectories.
        output_dir: Directory to save visualizations. If None, displays interactively.
        device: Device to run models on.

    Returns:
        Dict mapping model names to their statistics dict.
    """
    dataset_path = Path(dataset_path).expanduser()
    records = json.loads(dataset_path.read_text())
    puzzles = [r for r in records if r.get('step', 0) == 0]
    logger.info(f"Loaded {len(puzzles)} step-0 puzzles from {dataset_path.name}")

    models_path = Path(models_dir)

    analyzers = {}
    for key, config in MODEL_CONFIGS.items():
        model_path = models_path / config['subdir'] / 'best_model.pt'
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}, skipping {config['display_name']}")
            continue
        logger.info(f"Loading {config['display_name']} from {model_path}")
        analyzers[config['display_name']] = FailureAnalyzer(model_path=str(model_path), device=device)

    if not analyzers:
        raise ValueError(f"No models found in {models_dir}")

    print(f"Evaluating {len(analyzers)} models on {len(puzzles)} puzzles...")

    for puzzle_idx, puzzle in enumerate(puzzles):
        ground_truth = Solver.solve_with_vanilla_backtracking(puzzle)

        for name, analyzer in analyzers.items():
            analyzer.model.to(device)
            analyzer.analyze_puzzle(puzzle, ground_truth=ground_truth)
            analyzer.model.to('cpu')

        if (puzzle_idx + 1) % 25 == 0:
            accs = {n: f"{a.get_accuracy():.1f}%" for n, a in analyzers.items()}
            print(f"  [{puzzle_idx + 1}/{len(puzzles)}] {accs}")

    torch.cuda.empty_cache() if device == 'cuda' else None
    print("Evaluation complete.")

    stats = {name: analyzer.get_stats() for name, analyzer in analyzers.items()}
    visualize_failure_statistics(stats, output_dir)

    return stats


def visualize_failure_statistics(
    stats: Dict[str, dict],
    output_dir: Optional[str] = None
) -> None:
    """
    Visualize failure statistics across models.

    Creates three plots:
    1. Grouped bar chart of puzzle sizes that failed
    2. Grouped bar chart of step number for first mistake
    3. Box plot of logit confidence on first mistake
    """
    model_names = list(stats.keys())
    num_models = len(model_names)
    colors = plt.cm.tab10(np.arange(num_models))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    all_sizes = []
    for name in model_names:
        all_sizes.extend(stats[name].get('all_sizes', stats[name]['failed_sizes']))
    if all_sizes:
        min_size = min(all_sizes)
        max_size = max(all_sizes)
        size_values = list(range(min_size, max_size + 1))
    else:
        size_values = list(range(5, 12))

    x = np.arange(len(size_values))
    width = 0.8 / num_models

    for idx, name in enumerate(model_names):
        failed = stats[name]['failed_sizes']
        evaluated = stats[name].get('all_sizes', [])
        total_per_size = {s: evaluated.count(s) for s in size_values}
        failed_per_size = {s: failed.count(s) for s in size_values}
        rates = [
            failed_per_size[s] / total_per_size[s] * 100 if total_per_size[s] > 0 else 0
            for s in size_values
        ]
        offset = (idx - num_models / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=name, color=colors[idx], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Puzzle Size (n)', fontsize=12)
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_title('Failure Rate by Puzzle Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_values)
    ax.legend(loc='upper right')

    ax = axes[1]
    all_steps = []
    for name in model_names:
        all_steps.extend(stats[name]['first_mistake_steps'])
    if all_steps:
        max_step = max(all_steps)
        step_values = list(range(1, max_step + 1))
    else:
        step_values = list(range(1, 12))

    x = np.arange(len(step_values))
    width = 0.8 / num_models

    for idx, name in enumerate(model_names):
        steps = stats[name]['first_mistake_steps']
        counts = [steps.count(s) for s in step_values]
        offset = (idx - num_models / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=name, color=colors[idx], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Step Number of First Mistake', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('When Models Make First Mistake', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(step_values)
    ax.legend(loc='upper right')

    ax = axes[2]
    logit_data = []
    labels = []
    valid_colors = []
    for idx, name in enumerate(model_names):
        logits = stats[name]['first_mistake_logits']
        if logits:
            logit_data.append(logits)
            labels.append(name)
            valid_colors.append(colors[idx])

    if logit_data:
        bp = ax.boxplot(logit_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha='right')

    ax.set_ylabel('Logit Confidence', fontsize=12)
    ax.set_title('Confidence on First Mistake', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / "failure_statistics.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {filename}")
        plt.close()

        size_dist_path = output_path / "puzzle_size_distribution.txt"
        with open(size_dist_path, 'w') as f:
            f.write("Puzzle Size Distribution\n")
            f.write("=" * 60 + "\n\n")
            ref_name = model_names[0]
            ref_sizes = stats[ref_name].get('all_sizes', [])
            f.write(f"{'Size':>6} {'Count':>8} {'% of Total':>12}\n")
            f.write("-" * 30 + "\n")
            total = len(ref_sizes)
            for s in size_values:
                count = ref_sizes.count(s)
                pct = count / total * 100 if total > 0 else 0
                f.write(f"{s:>6} {count:>8} {pct:>11.1f}%\n")
            f.write(f"{'Total':>6} {total:>8}\n")

            f.write("\n\nPer-Model Failure Rates by Size\n")
            f.write("=" * 60 + "\n\n")
            header = f"{'Size':>6}"
            for name in model_names:
                header += f" {name:>16}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for s in size_values:
                line = f"{s:>6}"
                for name in model_names:
                    evaluated = stats[name].get('all_sizes', [])
                    failed = stats[name]['failed_sizes']
                    t = evaluated.count(s)
                    fc = failed.count(s)
                    rate = fc / t * 100 if t > 0 else 0
                    line += f" {fc}/{t} ({rate:.0f}%)".rjust(16)
                f.write(line + "\n")
        print(f"Saved: {size_dist_path}")
    else:
        plt.show()

    print("\n" + "=" * 60)
    print("FAILURE STATISTICS SUMMARY")
    print("=" * 60)
    for name in model_names:
        s = stats[name]
        accuracy = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"\n{name}:")
        print(f"  Accuracy: {s['correct']}/{s['total']} ({accuracy:.1f}%)")

        if s['first_mistake_logits']:
            logits = s['first_mistake_logits']
            print(f"  First mistake logit - Median: {np.median(logits):.3f}, "
                  f"IQR: [{np.percentile(logits, 25):.3f}, {np.percentile(logits, 75):.3f}]")

        if s['first_mistake_steps']:
            steps = s['first_mistake_steps']
            print(f"  First mistake step - Median: {np.median(steps):.1f}, "
                  f"IQR: [{np.percentile(steps, 25):.1f}, {np.percentile(steps, 75):.1f}]")
    print("=" * 60)


def visualize_model_comparison(
    puzzle: dict,
    models_dir: str = 'checkpoints/app_models',
    output_dir: Optional[str] = None,
    show_activation: bool = True,
    device: str = 'cuda',
    hrm_cycle: int = 1,
    include_benchmarks: bool = True
) -> Dict[str, dict]:
    """
    Visualize step-by-step comparison of multiple models solving the same puzzle.

    Creates one figure per placement step, with each model as a row showing:
    Board State | Final Attention | Activation (optional) | Logits

    Args:
        puzzle: Dict with 'region' key containing the region board.
        models_dir: Path to directory containing model subdirectories.
        output_dir: Directory to save images. If None, displays interactively.
        show_activation: If True, show activation column (H-late for HRM, empty for others).
        device: Device to run models on.
        hrm_cycle: Which HRM attention cycle to visualize (1, 2, or 3).
        include_benchmarks: If True, include benchmark models in the comparison.

    Returns:
        Dict mapping model names to their diagnostic results.
    """
    models_path = Path(models_dir)
    ground_truth = Solver.solve_with_vanilla_backtracking(puzzle)

    model_results = {}
    model_order = []

    for key, config in MODEL_CONFIGS.items():
        if not include_benchmarks and key.startswith('benchmark_'):
            continue
        model_path = models_path / config['subdir'] / 'best_model.pt'
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}, skipping {config['display_name']}")
            continue

        logger.info(f"Loading {config['display_name']} from {model_path}")
        analyzer = FailureAnalyzer(model_path=str(model_path), device=device)

        result = analyzer.solve_puzzle_diagnostic(
            puzzle,
            ground_truth=ground_truth,
            verbose=False,
            capture_attention=True,
            capture_activations=config['is_hrm'],
            batch_placement=False
        )

        model_results[config['display_name']] = result
        model_order.append(config['display_name'])

    if not model_results:
        raise ValueError(f"No models found in {models_dir}")

    _generate_comparison_figures(
        puzzle, model_results, model_order, ground_truth, output_dir, show_activation, hrm_cycle
    )

    return model_results


def _generate_comparison_figures(
    puzzle: dict,
    model_results: Dict[str, dict],
    model_order: List[str],
    ground_truth: np.ndarray,
    output_dir: Optional[str],
    show_activation: bool,
    hrm_cycle: int = 1
) -> None:
    """Generate the actual comparison figures."""
    region_board = np.array(puzzle['region'])
    n = region_board.shape[0]
    correct_positions = set(zip(*np.where(ground_truth == 1)))

    num_models = len(model_order)
    num_steps = n
    num_cols = 4 if show_activation else 3

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    model_placed_queens = {name: [] for name in model_order}

    for step in range(num_steps):
        fig, axes = plt.subplots(num_models, num_cols, figsize=(6 * num_cols, 5 * num_models))
        if num_models == 1:
            axes = [axes]

        fig.suptitle(f'Step {step + 1}/{num_steps}', fontsize=16, fontweight='bold')

        for row_idx, model_name in enumerate(model_order):
            result = model_results[model_name]
            activations = result.get('activations', [])
            passes = result.get('passes', [])

            if activations and step < len(activations) and activations[step] is not None:
                step_attention = activations[step]
            else:
                step_attention = {}

            if step < len(passes):
                step_pass = passes[step]
                placement = step_pass['queens_placed'][0] if step_pass['queens_placed'] else (0, 0)
                pred_logit = step_pass['logits_placed'][0] if step_pass['logits_placed'] else 0.0
                first_correct_rank = step_pass.get('first_correct_rank')
                top_k_preds = step_pass.get('top_k_predictions', [])
            else:
                placement = (0, 0)
                pred_logit = 0.0
                first_correct_rank = None
                top_k_preds = []

            placed_queens = model_placed_queens[model_name]
            pred_row, pred_col = placement

            remaining_correct = correct_positions - set(placed_queens)
            is_correct = placement in remaining_correct

            correct_logit = None
            if not is_correct and first_correct_rank is not None:
                for pred in top_k_preds:
                    if pred['is_correct']:
                        correct_logit = pred['logit']
                        break

            status = "Y" if is_correct else "N"
            info_line = f"->({pred_row},{pred_col})  logit: {pred_logit:.2f}"
            if not is_correct and first_correct_rank is not None:
                if correct_logit is not None:
                    info_line += f"\ncorrect: rank {first_correct_rank}, logit: {correct_logit:.2f}"
                else:
                    info_line += f"\ncorrect: rank {first_correct_rank}"
            elif not is_correct:
                info_line += "\ncorrect: >top-k"

            col_idx = 0

            ax = axes[row_idx][col_idx]
            _draw_board_for_comparison(ax, region_board, placed_queens, pred_row, pred_col, is_correct)

            board_title = f"{model_name} {status}\n{info_line}"
            ax.set_title(board_title, fontsize=10, fontweight='bold', loc='left')
            col_idx += 1

            ax = axes[row_idx][col_idx]
            attn_map = _extract_final_attention(step_attention, n, pred_row, pred_col, hrm_cycle)
            if attn_map is not None:
                im = ax.imshow(attn_map, cmap='viridis')
                ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.5)
                for qr, qc in placed_queens:
                    ax.text(qc, qr, 'X', fontsize=14, ha='center', va='center',
                           color='white', fontweight='bold')
                _draw_prediction_marker(ax, pred_col, pred_row, is_correct)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                       transform=ax.transAxes, color='gray')
                ax.set_facecolor('#f0f0f0')
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title('Final Attention', fontsize=11, fontweight='bold')
            col_idx += 1

            if show_activation:
                ax = axes[row_idx][col_idx]
                act_map = _extract_h_activation(step_attention, n)
                if act_map is not None:
                    im = ax.imshow(act_map, cmap='PuOr_r', vmin=0, vmax=1)
                    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                    ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.5)
                    for qr, qc in placed_queens:
                        ax.text(qc, qr, 'X', fontsize=14, ha='center', va='center',
                               color='white', fontweight='bold')
                    _draw_prediction_marker(ax, pred_col, pred_row, is_correct)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                           transform=ax.transAxes, color='gray')
                    ax.set_facecolor('#f0f0f0')
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title('H Activation', fontsize=11, fontweight='bold')
                col_idx += 1

            ax = axes[row_idx][col_idx]
            logits = step_attention.get('logits', np.zeros((n, n)))
            im = ax.imshow(logits, cmap='RdBu_r')
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which='minor', color='black', linewidth=0.5)
            for qr, qc in placed_queens:
                ax.text(qc, qr, 'X', fontsize=14, ha='center', va='center',
                       color='black', fontweight='bold')

            _draw_prediction_marker(ax, pred_col, pred_row, is_correct, on_logits=True)

            if not is_correct and first_correct_rank is not None:
                for pred in top_k_preds:
                    if pred['is_correct']:
                        correct_pos = pred['position']
                        ax.scatter([correct_pos[1]], [correct_pos[0]], marker='o', facecolors='none',
                                  edgecolors='lime', s=400, linewidths=3, zorder=9)
                        break

            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title('Logits', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            model_placed_queens[model_name].append(placement)

        plt.tight_layout()

        if output_dir:
            filename = output_path / f"step_{step:02d}_comparison.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def _draw_board_for_comparison(ax, region_board: np.ndarray, placed_queens: List[Tuple[int, int]],
                                pred_row: int, pred_col: int, is_correct: bool) -> None:
    """Draw board with colored regions and placed queens for comparison view."""
    n = region_board.shape[0]
    num_regions = region_board.max() + 1

    cmap = plt.cm.get_cmap('tab20', num_regions)
    board_colors = np.zeros((n, n, 4))
    for i in range(n):
        for j in range(n):
            region_id = region_board[i, j]
            board_colors[i, j] = cmap(region_id)

    ax.imshow(board_colors)

    for i in range(n + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    for qr, qc in placed_queens:
        ax.text(qc, qr, 'Q', fontsize=16, ha='center', va='center',
               color='black', fontweight='bold')

    _draw_prediction_marker(ax, pred_col, pred_row, is_correct)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_prediction_marker(ax, col: int, row: int, is_correct: bool, on_logits: bool = False) -> None:
    """Draw prediction marker - green star if correct, red star if wrong."""
    if is_correct:
        color = 'lime'
        ax.scatter([col], [row], marker='*', color=color, s=500,
                  edgecolors='white', linewidths=2, zorder=10)
    else:
        ax.scatter([col], [row], marker='*', color='red', s=500,
                  edgecolors='white', linewidths=2, zorder=10)


def _extract_final_attention(attention_data: dict, n: int, query_row: int, query_col: int,
                              hrm_cycle: int = 1) -> Optional[np.ndarray]:
    """Extract attention map for visualization."""
    query_idx = query_row * n + query_col

    if 'H_attentions' in attention_data:
        H_attentions = attention_data['H_attentions']
        if H_attentions:
            cycle_idx = min(hrm_cycle - 1, len(H_attentions) - 1)
            cycle_idx = max(0, cycle_idx)
            selected_attn = H_attentions[cycle_idx]
            attn_from_query = selected_attn[0, :, query_idx, :].mean(dim=0).numpy()
            return attn_from_query.reshape(n, n)

    if 'gat_attention' in attention_data:
        gat_attention = attention_data['gat_attention']

        if isinstance(gat_attention, list):
            if gat_attention:
                last_layer = gat_attention[-1]
                edge_index = last_layer['edge_index']
                alpha = last_layer['alpha']

                attn_map = np.zeros(n * n)
                mask = edge_index[0] == query_idx
                if mask.any():
                    targets = edge_index[1][mask].numpy()
                    weights = alpha[mask].mean(dim=-1).numpy()
                    for t, w in zip(targets, weights):
                        attn_map[t] = w
                return attn_map.reshape(n, n)

        else:
            last_layer_idx = max(gat_attention.keys())
            layer_attn = gat_attention[last_layer_idx]
            edge_types = list(layer_attn.keys())

            combined_attn_map = np.zeros(n * n)
            for edge_type in edge_types:
                edge_data = layer_attn[edge_type]
                edge_index = edge_data['edge_index']
                alpha = edge_data['alpha']

                mask = edge_index[0] == query_idx
                if mask.any():
                    targets = edge_index[1][mask].numpy()
                    weights = alpha[mask].mean(dim=-1).numpy()
                    for t, w in zip(targets, weights):
                        combined_attn_map[t] += w
            return combined_attn_map.reshape(n, n)

    return None


def _extract_h_activation(attention_data: dict, n: int) -> Optional[np.ndarray]:
    """Extract H activation map for HRM models only."""
    if 'H' in attention_data:
        H_maps = attention_data['H']
        if 'late' in H_maps:
            return H_maps['late']
        elif 'mid' in H_maps:
            return H_maps['mid']
        elif 'early' in H_maps:
            return H_maps['early']
        elif 'final' in H_maps:
            return H_maps['final']

    return None
