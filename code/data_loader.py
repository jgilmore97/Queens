from __future__ import annotations

import json
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset 
from torch_geometric.loader import DataLoader


class EdgeIndexCache:
    """Cache for pre-computed structural edges (rows, cols, diagonals) by board size."""
    
    def __init__(self):
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    
    def get_structural_edges(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get pre-computed row, column, and diagonal edges for board size n."""
        if n not in self._cache:
            self._cache[n] = self._compute_structural_edges(n)
        return self._cache[n]
    
    def _compute_structural_edges(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-compute all structural edges that don't depend on region layout."""
        node_idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
        
        # Row edges
        row_edges = []
        for r in range(n):
            row_nodes = node_idx[r, :]
            i, j = np.triu_indices(n, k=1)
            row_edges.append(np.column_stack([row_nodes[i], row_nodes[j]]))
        row_edges = np.vstack(row_edges)
        
        # Column edges
        col_edges = []
        for c in range(n):
            col_nodes = node_idx[:, c]
            i, j = np.triu_indices(n, k=1)
            col_edges.append(np.column_stack([col_nodes[i], col_nodes[j]]))
        col_edges = np.vstack(col_edges)
        
        # Diagonal edges
        # Main diagonal (↘)
        r, c = np.mgrid[0:n-1, 0:n-1]
        diag1 = np.column_stack([
            node_idx[r, c].ravel(),
            node_idx[r + 1, c + 1].ravel()
        ])
        
        # Anti-diagonal (↙)
        r, c = np.mgrid[0:n-1, 1:n]
        diag2 = np.column_stack([
            node_idx[r, c].ravel(),
            node_idx[r + 1, c - 1].ravel()
        ])
        
        diag_edges = np.vstack([diag1, diag2])
        
        return row_edges, col_edges, diag_edges


def _build_edge_index_optimized(region: np.ndarray, structural_cache: EdgeIndexCache) -> torch.Tensor:
    """
    Highly optimized edge index creation using vectorized operations.
    
    Returns undirected edge_index tensor capturing Queens constraints.
    """
    n = region.shape[0]
    
    # Get pre-computed structural edges from provided cache
    row_edges, col_edges, diag_edges = structural_cache.get_structural_edges(n)
    
    # Compute region edges (the only part that varies by board)
    region_edges = _compute_region_edges_vectorized(region)
    
    # Stack all edges
    all_edges = np.vstack([row_edges, col_edges, diag_edges, region_edges])
    
    # Add reverse edges for undirected graph
    edges_undirected = np.vstack([all_edges, all_edges[:, [1, 0]]])
    
    return torch.tensor(edges_undirected.T, dtype=torch.long).contiguous()


def _compute_region_edges_vectorized(region: np.ndarray) -> np.ndarray:
    """
    Compute region edges using fully vectorized operations.
    
    This is the only part that changes between boards, so we optimize it separately.
    """
    n = region.shape[0]
    flat_region = region.ravel()
    
    # Use unique with return_inverse for efficient grouping
    unique_regions, inverse = np.unique(flat_region, return_inverse=True)
    
    # Pre-allocate list for edge arrays
    edge_arrays = []
    
    for region_id in range(len(unique_regions)):
        # Get all cells in this region using boolean mask
        region_mask = (inverse == region_id)
        region_cells = np.where(region_mask)[0]
        
        n_cells = len(region_cells)
        if n_cells > 1:
            # Generate all pairs using triangular indices
            i, j = np.triu_indices(n_cells, k=1)
            edges = np.column_stack([region_cells[i], region_cells[j]])
            edge_arrays.append(edges)
    
    # Combine all region edges
    if edge_arrays:
        return np.vstack(edge_arrays)
    else:
        return np.empty((0, 2), dtype=np.int64)


# def _build_edge_index(region: np.ndarray) -> torch.Tensor:
#     """Return undirected edge_index tensor capturing Queens constraints."""
#     n = region.shape[0]
#     idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
#     edges: list[Tuple[int, int]] = []

#     # rows
#     for r in range(n):
#         for i, j in combinations(idx[r, :], 2):
#             edges += [(i, j), (j, i)]

#     # columns
#     for c in range(n):
#         for i, j in combinations(idx[:, c], 2):
#             edges += [(i, j), (j, i)]

#     # regions
#     for reg in np.unique(region):
#         nodes = idx[region == reg].ravel()
#         for i, j in combinations(nodes, 2):
#             edges += [(i, j), (j, i)]

#     # immediate diagonals
#     for r in range(n - 1):
#         for c in range(n - 1):
#             a, b = idx[r, c], idx[r + 1, c + 1]  # ↘
#             edges += [(a, b), (b, a)]
#         for c in range(1, n):
#             a, b = idx[r, c], idx[r + 1, c - 1]  # ↙
#             edges += [(a, b), (b, a)]

#     return torch.tensor(edges, dtype=torch.long).t().contiguous()


#--------------------------------------------------------------
# 2.  Utility – deterministic group‐aware splitter
#--------------------------------------------------------------

def _canonical_img_key(src: str) -> str:
    """IMG_5193.jpg_rot0  →  IMG_5193.jpg"""

    _IMG_RE = re.compile(r"^(.*?\.jpg)", re.IGNORECASE)

    m = _IMG_RE.match(src)
    if not m:
        raise ValueError(f"Could not parse image key from source='{src}'")
    return m.group(1)


def _split_by_img(
    records: List[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Group records by photo, then split groups into train / val."""
    # --- group
    groups: Dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(_canonical_img_key(rec["source"]), []).append(rec)

    # --- shuffle group keys deterministically
    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    # --- allocate to val / train
    n_val = int(len(keys) * val_ratio)
    val_keys = set(keys[:n_val])

    train, val = [], []
    for k, recs in groups.items():
        (val if k in val_keys else train).extend(recs)

    return train, val


class QueensDataset(Dataset):
    """
    PyTorch Geometric Dataset for the Queens puzzle.
    - Region IDs are one-hot encoded and padded to the largest board in the JSON.
    - Row/col coordinates are min-max scaled to the [0 , 1] range.
    """

    # Class-level caches shared across all instances
    _split_cache: Dict[tuple[Path, float, int], Tuple[list[dict], list[dict]]] = {}
    _structural_edge_cache = EdgeIndexCache()  # Shared structural edge cache

    def __init__(
        self,
        json_path: str | Path,
        *,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        transform=None,
        pre_transform=None,
        cache_edges: bool = True,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")

        super().__init__(None, transform, pre_transform)

        self.json_path   = Path(json_path).expanduser()
        self.val_ratio   = val_ratio
        self.seed        = seed
        self.cache_edges = cache_edges
        self._board_edge_cache: dict[int, torch.Tensor] = {}  # Instance cache for complete edge indices

        # ------------------------------------------------------
        # 1) build / fetch cached train-val split
        # ------------------------------------------------------
        key = (self.json_path, val_ratio, seed)
        if key not in self._split_cache:
            records          = json.loads(self.json_path.read_text())
            train, val       = _split_by_img(records, val_ratio, seed)
            self._split_cache[key] = (train, val)

        train_set, val_set = self._split_cache[key]

        # ------------------------------------------------------
        # 2) expose requested subset
        # ------------------------------------------------------
        self.records = (
            train_set if split == "train"
            else val_set if split == "val"
            else train_set + val_set
        )

        # ------------------------------------------------------
        # 3) largest region count in the whole JSON → padding size
        # ------------------------------------------------------
        self.max_regions = max(
            int(np.max(r["region"])) + 1
            for r in (train_set + val_set)
        )

    # ------------------------------------------------------------------
    # PyG Dataset hooks
    # ------------------------------------------------------------------
    def len(self) -> int:  # noqa: D401
        return len(self.records)

    def get(self, idx: int) -> Data:  # noqa: D401
        e = self.records[idx]

        region  = np.asarray(e["region"],        dtype=np.int64)   # (n, n)
        partial = np.asarray(e["partial_board"], dtype=np.int64)   # (n, n)
        label   = np.asarray(e["label_board"],   dtype=np.int64)   # (n, n)
        n       = region.shape[0]
        N2      = n * n

        # --- node features ---------------------------------------------
        # 1) scaled coordinates
        coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)  # (N², 2)

        # 2) padded one-hot region
        reg_onehot = np.zeros((N2, self.max_regions), dtype=np.float32)
        flat_ids   = region.flatten()
        reg_onehot[np.arange(N2), flat_ids] = 1.0                                  # (N², R)

        # 3) has-queen flag
        has_q = partial.flatten()[:, None].astype(np.float32)                      # (N², 1)

        x = torch.from_numpy(np.hstack([coords, reg_onehot, has_q]))               # (N², 3+R)

        # --- label -------------------------------------------------------
        y = torch.from_numpy(label.flatten().astype(np.int64))                     # (N²,)

        # --- edge_index (with optional cache) ----------------------------
        board_key = hash(region.tobytes()) if self.cache_edges else None
        if board_key is not None and board_key in self._board_edge_cache:
            edge_index = self._board_edge_cache[board_key]
        else:
            # Use the class-level structural cache
            edge_index = _build_edge_index_optimized(region, self._structural_edge_cache)
            if board_key is not None:
                self._board_edge_cache[board_key] = edge_index

        # --- assemble Data object ---------------------------------------
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            n=torch.tensor([n], dtype=torch.long),
            step=torch.tensor([e["step"]], dtype=torch.long),
            meta=dict(source=e["source"], iteration=e["iteration"]),
        )
    

def get_queens_loaders(
    json_path: str,
    *,
    batch_size: int = 512,
    val_ratio: float = 0.10,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    follow_batch: list[str] | None = None,
    shuffle_train: bool = True,
):
    """
    Return (train_loader, val_loader).

    Parameters
    ----------
    json_path : str
        Path to the *master* JSON file (non-test records).
    batch_size : int, default 32
        Batch size for both loaders.
    val_ratio : float, default 0.20
        Fraction of distinct image sources that go into validation.
        Must match between train and val loaders.
    seed : int, default 42
        Random seed for deterministic split.
    num_workers : int, default 0
        Passed straight to `torch_geometric.loader.DataLoader`.
    pin_memory : bool, default True
        Pin memory for faster host-to-GPU transfers.
    follow_batch : list[str] | None
        Optional list of attribute names to generate *_batch vectors
        (e.g. ["x"] if you need node-wise batch IDs).
    shuffle_train : bool, default True
        Whether to shuffle the training loader each epoch.
    """
    # -------- datasets ----------------------------------------
    ds_train = QueensDataset(
        json_path,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
    )
    ds_val = QueensDataset(
        json_path,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )

    # -------- loaders -----------------------------------------
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        follow_batch=follow_batch or [],
    )

    train_loader = DataLoader(ds_train, shuffle=shuffle_train, **kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False,        **kwargs)

    return train_loader, val_loader