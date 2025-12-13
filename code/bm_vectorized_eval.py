"""
Vectorized full-puzzle evaluation for benchmark transformer model.
Runs batched autoregressive solve across multiple puzzles simultaneously.
"""

import torch
import numpy as np
from typing import Dict, Tuple

from data_loader import BenchmarkDataset


class BenchmarkStep0Dataset(BenchmarkDataset):
    """Dataset filtered to only step-0 (empty board) puzzles for evaluation."""

    def __init__(self, json_path: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__(
            json_path,
            split="all",
            val_ratio=val_ratio,
            seed=seed
        )
        # Filter to step-0 only
        self.records = [r for r in self.records if r.get('step', 0) == 0]
        print(f"BenchmarkStep0Dataset: {len(self.records)} step-0 puzzles")

    def build_features_with_queens(self, idx: int, queen_board: np.ndarray) -> torch.Tensor:
        """
        Build features for a puzzle with a custom queen board state.
        Reuses the feature encoding logic from BenchmarkDataset.
        """
        e = self.records[idx]
        region = np.asarray(e["region"], dtype=np.int64)

        # Pad region and queen_board
        region_padded = self.pad(region, target_size=self.max_regions, pad_with=-1)
        queen_padded = self.pad(queen_board.astype(np.int64), target_size=self.max_regions, pad_with=0)

        # Normalized coordinates
        coords = np.indices((self.max_regions, self.max_regions)).reshape(2, -1).T.astype(np.float32) / (self.max_regions - 1)

        # Region one-hot
        reg_onehot = np.zeros((self.max_regions * self.max_regions, self.max_regions), dtype=np.float32)
        flat_ids = region_padded.flatten()
        valid_mask = flat_ids != -1
        reg_onehot[np.arange(self.max_regions * self.max_regions)[valid_mask], flat_ids[valid_mask]] = 1.0

        # Has queen indicator
        has_q = queen_padded.flatten()[:, None].astype(np.float32)

        x = np.hstack([coords, reg_onehot, has_q])
        return torch.from_numpy(x)


def evaluate_solve_rate(
    model: torch.nn.Module,
    json_path: str,
    device: str,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate model's full-puzzle solve rate using batched autoregressive inference.
    All puzzles are padded to same size so no grouping needed.
    """
    model.eval()

    dataset = BenchmarkStep0Dataset(json_path, val_ratio=val_ratio, seed=seed)
    max_size = dataset.max_regions

    total_puzzles = 0
    solved_puzzles = 0
    error_by_step = {}

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_indices = list(range(batch_start, batch_end))

            batch_solved, batch_total, batch_errors = _evaluate_batch(
                model, dataset, batch_indices, device, max_size
            )

            solved_puzzles += batch_solved
            total_puzzles += batch_total

            for step, count in batch_errors.items():
                error_by_step[step] = error_by_step.get(step, 0) + count

    solve_rate = solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0

    return {
        'solve_rate': solve_rate,
        'total_puzzles': total_puzzles,
        'solved_puzzles': solved_puzzles,
        'error_by_step': error_by_step
    }


def _evaluate_batch(
    model: torch.nn.Module,
    dataset: BenchmarkStep0Dataset,
    batch_indices: list,
    device: str,
    max_size: int = 11
) -> Tuple[int, int, Dict[int, int]]:
    """
    Evaluate a single batch of puzzles autoregressively.
    """
    num_puzzles = len(batch_indices)

    # Get board sizes and initialize queen boards
    board_sizes = []
    queen_boards = []
    for idx in batch_indices:
        n = len(dataset.records[idx]['region'])
        board_sizes.append(n)
        queen_boards.append(np.zeros((max_size, max_size), dtype=np.int64))

    max_steps = max(board_sizes)

    # Extract label boards and create valid masks
    labels_flat = []
    for idx in batch_indices:
        record = dataset.records[idx]
        label = np.asarray(record["label_board"], dtype=np.int64)
        label_padded = dataset.pad(label, target_size=max_size, pad_with=-1)
        labels_flat.append(label_padded.flatten())
    labels_flat = torch.from_numpy(np.stack(labels_flat)).to(device)

    valid_mask_tensor = (labels_flat >= 0)

    # Track solve status
    still_correct = torch.ones(num_puzzles, dtype=torch.bool, device=device)
    first_error_step = torch.full((num_puzzles,), -1, dtype=torch.long, device=device)
    placed = torch.zeros((num_puzzles, max_size * max_size), dtype=torch.bool, device=device)

    # Track which puzzles are done (reached their n steps)
    steps_remaining = torch.tensor(board_sizes, device=device)

    for step in range(max_steps):
        # Skip puzzles that have completed all their steps
        active = steps_remaining > step

        # Build features
        x_list = [dataset.build_features_with_queens(idx, queen_boards[i])
                  for i, idx in enumerate(batch_indices)]
        x = torch.stack(x_list).to(device)

        # Forward pass
        logits = model(x).squeeze(-1)

        # Mask out already placed and invalid positions
        masked_logits = logits.clone()
        masked_logits[placed] = float('-inf')
        masked_logits[~valid_mask_tensor] = float('-inf')

        # Select best position for each puzzle
        selected_indices = masked_logits.argmax(dim=1)

        # Check correctness only for active puzzles
        selected_labels = labels_flat[torch.arange(num_puzzles, device=device), selected_indices]
        is_correct = (selected_labels == 1) | ~active

        # Track first error
        just_failed = still_correct & ~is_correct & active
        first_error_step[just_failed] = step
        still_correct = still_correct & is_correct

        # Update placed mask
        placed[torch.arange(num_puzzles, device=device), selected_indices] = True

        # Update queen_boards for next iteration
        selected_rows = (selected_indices // max_size).cpu().numpy()
        selected_cols = (selected_indices % max_size).cpu().numpy()
        for i in range(num_puzzles):
            if active[i]:
                queen_boards[i][selected_rows[i], selected_cols[i]] = 1

    solved_count = still_correct.sum().item()

    # Collect error distribution
    error_by_step = {}
    for step in range(max_steps):
        count = (first_error_step == step).sum().item()
        if count > 0:
            error_by_step[step] = count

    return solved_count, num_puzzles, error_by_step


def print_results(results: Dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("BENCHMARK MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total puzzles: {results['total_puzzles']}")
    print(f"Solved puzzles: {results['solved_puzzles']}")
    print(f"Solve rate: {results['solve_rate']:.1%}")

    if results['error_by_step']:
        print("\nFirst error distribution by step:")
        failed = results['total_puzzles'] - results['solved_puzzles']
        for step in sorted(results['error_by_step'].keys()):
            count = results['error_by_step'][step]
            pct = count / failed if failed > 0 else 0
            print(f"  Step {step}: {count} errors ({pct:.1%} of failures)")


if __name__ == "__main__":
    import argparse
    from bm_model import BenchmarkComparisonModel

    parser = argparse.ArgumentParser(description="Evaluate benchmark model on Queens puzzles")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = BenchmarkComparisonModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    # Evaluate
    results = evaluate_solve_rate(
        model,
        args.data,
        args.device,
        batch_size=args.batch_size
    )

    print_results(results)
