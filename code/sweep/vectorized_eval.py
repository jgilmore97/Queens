import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch_geometric.data import Batch
from pathlib import Path

from data_loader import QueensDataset


@dataclass
class FailedPuzzle:
    """Record of a puzzle that failed during evaluation."""
    dataset_idx: int
    source: str
    board_size: int
    first_error_step: int
    region: np.ndarray
    label_board: np.ndarray
    predicted_positions: List[Tuple[int, int]]
    correct_positions: List[Tuple[int, int]]


class Step0Dataset(QueensDataset):
    """Dataset filtered to step-0 (empty board) puzzles only."""

    def __init__(self, json_path: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__(
            json_path,
            split="all",
            val_ratio=val_ratio,
            seed=seed
        )
        self.records = [r for r in self.records if r.get('step', 0) == 0]
        print(f"Step0Dataset: {len(self.records)} step-0 puzzles")


class _MutableBatch:
    """
    Wrapper providing mutable view of batch data for autoregressive inference.
    Presents the interface expected by HRM.forward().
    """
    def __init__(self, batch, x_cell: torch.Tensor):
        self._batch = batch
        self._x_cell = x_cell

    @property
    def x_dict(self) -> Dict[str, torch.Tensor]:
        return {'cell': self._x_cell}

    @property
    def edge_index_dict(self) -> Dict[Tuple[str, str, str], torch.Tensor]:
        return {
            ('cell', 'line_constraint', 'cell'): self._batch[('cell', 'line_constraint', 'cell')].edge_index,
            ('cell', 'region_constraint', 'cell'): self._batch[('cell', 'region_constraint', 'cell')].edge_index,
            ('cell', 'diagonal_constraint', 'cell'): self._batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
        }

    @property
    def num_graphs(self) -> int:
        return self._batch.num_graphs

    def __getitem__(self, key):
        """Allow indexing for edge access."""
        return self._batch[key]

    def update_x(self, x_cell: torch.Tensor):
        """Update node features for next autoregressive step."""
        self._x_cell = x_cell


def evaluate_solve_rate(
    model: torch.nn.Module,
    val_json_path: str,
    device: str,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    """
    Evaluate full-puzzle solve rate using batched autoregressive inference.
    Groups puzzles by size for efficient batching.
    
    Returns dict with solve_rate, counts, error distribution, and list of failed puzzles.
    """
    model.eval()

    dataset = Step0Dataset(val_json_path, val_ratio=val_ratio, seed=seed)

    size_groups = {}
    for idx in range(len(dataset)):
        record = dataset.records[idx]
        n = len(record['region'])
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(idx)

    print(f"Puzzle sizes: {{{', '.join(f'{n}x{n}: {len(idxs)}' for n, idxs in sorted(size_groups.items()))}}}")

    total_puzzles = 0
    solved_puzzles = 0
    error_by_step = {}
    failed_puzzles: List[FailedPuzzle] = []

    with torch.no_grad():
        for n, indices in size_groups.items():
            for batch_start in range(0, len(indices), batch_size):
                batch_end = min(batch_start + batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]

                batch_data = [dataset.get(idx) for idx in batch_indices]
                batch = Batch.from_data_list(batch_data)
                batch = batch.to(device)

                batch_solved, batch_total, batch_errors, batch_failures = _evaluate_batch(
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


def _evaluate_batch(
    model: torch.nn.Module,
    batch,
    device: str,
    n: int,
    dataset: Step0Dataset,
    dataset_indices: List[int]
) -> Tuple[int, int, Dict[int, int], List[FailedPuzzle]]:
    """
    Evaluate a batch of same-sized puzzles autoregressively.
    Returns (solved_count, total_count, error_by_step, failed_puzzles).
    """
    x = batch['cell'].x.clone()
    y = batch['cell'].y
    batch_indices = batch['cell'].batch
    num_graphs = batch.num_graphs

    mutable_batch = _MutableBatch(batch, x)

    still_correct = torch.ones(num_graphs, dtype=torch.bool, device=device)
    first_error_step = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
    placed = torch.zeros_like(y, dtype=torch.bool)

    all_predicted_positions = [[] for _ in range(num_graphs)]

    for step in range(n):
        logits = model(mutable_batch)

        masked_logits = logits.clone()
        masked_logits[placed] = float('-inf')

        selected_nodes = _batched_argmax(masked_logits, batch_indices, num_graphs)

        # Convert selected nodes to (row, col) positions
        nodes_per_graph = n * n
        for graph_idx in range(num_graphs):
            local_node = selected_nodes[graph_idx].item() - graph_idx * nodes_per_graph
            row, col = local_node // n, local_node % n
            all_predicted_positions[graph_idx].append((row, col))

        selected_labels = y[selected_nodes]
        is_correct = (selected_labels == 1)

        just_failed = still_correct & ~is_correct
        first_error_step[just_failed] = step
        still_correct = still_correct & is_correct

        placed[selected_nodes] = True
        x[selected_nodes, -1] = 1.0
        mutable_batch.update_x(x)

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

        region = np.array(record['region'])
        label_board = np.array(record['label_board'])
        correct_positions = [(r, c) for r in range(n) for c in range(n) if label_board[r, c] == 1]

        failed_puzzles.append(FailedPuzzle(
            dataset_idx=dataset_idx,
            source=record.get('source', f'puzzle_{dataset_idx}'),
            board_size=n,
            first_error_step=first_error_step[local_idx].item(),
            region=region,
            label_board=label_board,
            predicted_positions=all_predicted_positions[local_idx],
            correct_positions=correct_positions
        ))

    return solved_count, num_graphs, error_by_step, failed_puzzles


def _batched_argmax(
    logits: torch.Tensor,
    batch_indices: torch.Tensor,
    num_graphs: int
) -> torch.Tensor:
    """
    Compute per-graph argmax of logits.
    Returns tensor of global node indices (one per graph).
    """
    device = logits.device

    nodes_per_graph = torch.bincount(batch_indices, minlength=num_graphs)
    max_nodes = nodes_per_graph.max().item()

    logits_matrix = torch.full((num_graphs, max_nodes), float('-inf'), device=device)

    graph_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    graph_offsets[1:] = nodes_per_graph.cumsum(0)
    position_in_graph = torch.arange(len(batch_indices), device=device) - graph_offsets[batch_indices]

    logits_matrix[batch_indices, position_in_graph] = logits

    local_argmax = logits_matrix.argmax(dim=1)
    global_argmax = graph_offsets[:-1] + local_argmax

    return global_argmax