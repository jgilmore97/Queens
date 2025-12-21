import torch
import numpy as np
from typing import Dict, Tuple
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, HeteroData
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import QueensDataset


class Step0Dataset(QueensDataset):
    """Dataset filtered to only step-0 (empty board) puzzles."""

    def __init__(self, json_path: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__(
            json_path,
            split="all",
            val_ratio=val_ratio,
            seed=seed
        )
        self.records = [r for r in self.records if r.get('step', 0) == 0]
        print(f"Step0Dataset: {len(self.records)} step-0 puzzles")


def evaluate_solve_rate(
    model: torch.nn.Module,
    val_json_path: str,
    device: str,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate model's full-puzzle solve rate using batched autoregressive inference.
    Groups puzzles by size to handle variable board dimensions.
    """
    model.eval()

    dataset = Step0Dataset(val_json_path, val_ratio=val_ratio, seed=seed)

    # Group puzzles by size (required for batching since H-module needs uniform C per batch)
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

    with torch.no_grad():
        for n, indices in size_groups.items():
            subset_records = [dataset.records[i] for i in indices]

            for batch_start in range(0, len(subset_records), batch_size):
                batch_end = min(batch_start + batch_size, len(subset_records))
                batch_records = subset_records[batch_start:batch_end]

                batch_data = [dataset.get(indices[batch_start + i]) for i in range(len(batch_records))]
                batch = Batch.from_data_list(batch_data)
                batch = batch.to(device)

                batch_solved, batch_total, batch_errors = _evaluate_batch(model, batch, device, n)

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


class _MutableBatch:
    """
    Wrapper that presents a mutable view of batch data for autoregressive inference.
    Allows modifying node features while preserving the batch interface expected by HRM.
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

    def update_x(self, x_cell: torch.Tensor):
        """Update node features for next autoregressive step."""
        self._x_cell = x_cell


def _evaluate_batch(
    model: torch.nn.Module,
    batch,
    device: str,
    n: int
) -> Tuple[int, int, Dict[int, int]]:
    """
    Evaluate a single batch of puzzles autoregressively.
    All puzzles in batch must have the same size n.
    """
    x = batch['cell'].x.clone()
    y = batch['cell'].y
    batch_indices = batch['cell'].batch
    num_graphs = batch.num_graphs

    # Create mutable batch wrapper for autoregressive updates
    mutable_batch = _MutableBatch(batch, x)

    still_correct = torch.ones(num_graphs, dtype=torch.bool, device=device)
    first_error_step = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
    placed = torch.zeros_like(y, dtype=torch.bool)

    for step in range(n):
        logits = model(mutable_batch)

        masked_logits = logits.clone()
        masked_logits[placed] = float('-inf')

        selected_nodes = _batched_argmax(masked_logits, batch_indices, num_graphs)

        selected_labels = y[selected_nodes]
        is_correct = (selected_labels == 1)

        just_failed = still_correct & ~is_correct
        first_error_step[just_failed] = step
        still_correct = still_correct & is_correct

        # Update state for next step
        placed[selected_nodes] = True
        x[selected_nodes, -1] = 1.0
        mutable_batch.update_x(x)

    solved_count = still_correct.sum().item()
    total_count = num_graphs

    error_by_step = {}
    for step in range(n):
        count = (first_error_step == step).sum().item()
        if count > 0:
            error_by_step[step] = count

    return solved_count, total_count, error_by_step


def _batched_argmax(
    logits: torch.Tensor,
    batch_indices: torch.Tensor,
    num_graphs: int
) -> torch.Tensor:
    """
    Compute argmax of logits within each graph.
    Returns [num_graphs] tensor of selected node indices (global indices into logits).
    """
    device = logits.device

    nodes_per_graph = torch.bincount(batch_indices, minlength=num_graphs)
    max_nodes = nodes_per_graph.max().item()

    logits_matrix = torch.full((num_graphs, max_nodes), float('-inf'), device=device)

    # Compute position within each graph (vectorized)
    graph_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    graph_offsets[1:] = nodes_per_graph.cumsum(0)
    position_in_graph = torch.arange(len(batch_indices), device=device) - graph_offsets[batch_indices]

    logits_matrix[batch_indices, position_in_graph] = logits

    local_argmax = logits_matrix.argmax(dim=1)

    graph_starts = graph_offsets[:-1]
    global_argmax = graph_starts + local_argmax

    return global_argmax