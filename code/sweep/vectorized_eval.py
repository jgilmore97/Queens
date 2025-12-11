"""
Vectorized full-puzzle evaluation for hyperparameter sweep.
Runs batched autoregressive solve across multiple puzzles simultaneously.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from torch_geometric.loader import DataLoader
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import QueensDataset, build_heterogeneous_edge_index_cached


class Step0Dataset(QueensDataset):
    """Dataset filtered to only step-0 (empty board) puzzles."""

    def __init__(self, json_path: str, val_ratio: float = 0.1, seed: int = 42):
        super().__init__(
            json_path,
            split="all",  # Use all 720 puzzles, not just val split
            val_ratio=val_ratio,
            seed=seed
        )
        # Filter to step-0 only
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

    # Load step-0 puzzles
    dataset = Step0Dataset(val_json_path, val_ratio=val_ratio, seed=seed)
    
    # Group puzzles by size
    size_groups = {}
    for idx in range(len(dataset)):
        record = dataset.records[idx]
        n = len(record['region'])  # board size
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(idx)
    
    print(f"Puzzle sizes: {{{', '.join(f'{n}x{n}: {len(idxs)}' for n, idxs in sorted(size_groups.items()))}}}")

    total_puzzles = 0
    solved_puzzles = 0
    error_by_step = {}

    with torch.no_grad():
        for n, indices in size_groups.items():
            # Create subset dataset for this size
            subset_records = [dataset.records[i] for i in indices]
            
            # Process in batches
            for batch_start in range(0, len(subset_records), batch_size):
                batch_end = min(batch_start + batch_size, len(subset_records))
                batch_records = subset_records[batch_start:batch_end]
                
                # Build batch manually
                batch_data = [dataset.get(indices[batch_start + i]) for i in range(len(batch_records))]
                from torch_geometric.data import Batch
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


def _evaluate_batch(
    model: torch.nn.Module,
    batch,
    device: str,
    n: int  # Now passed explicitly - all puzzles in batch have this size
) -> Tuple[int, int, Dict[int, int]]:
    """
    Evaluate a single batch of puzzles autoregressively.
    All puzzles in batch must have the same size n.
    """
    # Extract batch info
    x = batch['cell'].x.clone()
    y = batch['cell'].y
    batch_indices = batch['cell'].batch

    # Build edge index dict for model
    edge_index_dict = {
        ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
        ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
        ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
    }

    num_graphs = batch_indices.max().item() + 1

    still_correct = torch.ones(num_graphs, dtype=torch.bool, device=device)
    first_error_step = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
    placed = torch.zeros_like(y, dtype=torch.bool)

    # Use the passed n (correct for all puzzles in this batch)
    for step in range(n):
        x_dict = {'cell': x}
        logits = model(x_dict, edge_index_dict)

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
    Args:
        logits: [total_nodes] tensor of logits
        batch_indices: [total_nodes] tensor of graph indices
        num_graphs: Number of graphs in batch
    Returns:
        [num_graphs] tensor of selected node indices (global indices into logits)
    """
    device = logits.device

    # Get max nodes per graph for padding
    nodes_per_graph = torch.bincount(batch_indices, minlength=num_graphs)
    max_nodes = nodes_per_graph.max().item()

    # Create padded matrix [num_graphs, max_nodes]
    logits_matrix = torch.full((num_graphs, max_nodes), float('-inf'), device=device)

    # Compute position within each graph
    position_in_graph = torch.zeros_like(batch_indices)
    for g in range(num_graphs):
        mask = (batch_indices == g)
        position_in_graph[mask] = torch.arange(mask.sum(), device=device)

    # Fill matrix
    logits_matrix[batch_indices, position_in_graph] = logits

    # Argmax within each graph (local index)
    local_argmax = logits_matrix.argmax(dim=1)  # [num_graphs]

    # Convert local index back to global node index
    # For each graph g, global_index = (start of graph g) + local_argmax[g]
    graph_starts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    graph_starts[1:] = nodes_per_graph[:-1].cumsum(0)

    global_argmax = graph_starts + local_argmax

    return global_argmax
