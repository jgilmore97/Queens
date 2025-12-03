from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset as vanillaDataset, DataLoader as vanillaDataLoader


# Edge index cache: maps region hash -> edge index dict
_edge_index_cache: Dict[str, Dict[str, torch.Tensor]] = {}


def _region_hash(region: np.ndarray) -> str:
    """Create a hash key from the region array."""
    return hashlib.md5(region.tobytes()).hexdigest()


def build_heterogeneous_edge_index(region: np.ndarray) -> Dict[str, torch.Tensor]:
    """Return dictionary of edge_index tensors for each constraint type.

    Vectorized implementation for improved performance.
    """
    n = region.shape[0]
    idx = np.arange(n * n, dtype=np.int64).reshape(n, n)

    # Down-right diagonals: (r, c) <-> (r+1, c+1)
    a_dr = idx[:-1, :-1].ravel()
    b_dr = idx[1:, 1:].ravel()
    # Down-left diagonals: (r, c) <-> (r+1, c-1)
    a_dl = idx[:-1, 1:].ravel()
    b_dl = idx[1:, :-1].ravel()
    # Both directions for each edge
    diag_src = np.concatenate([a_dr, b_dr, a_dl, b_dl])
    diag_tgt = np.concatenate([b_dr, a_dr, b_dl, a_dl])
    diagonal_edges = np.stack([diag_src, diag_tgt], axis=0)

    # Upper triangle indices for pairwise combinations within rows/columns
    i_idx, j_idx = np.triu_indices(n, k=1)

    # Row constraints: for each row r, connect idx[r, i] <-> idx[r, j]
    row_src = idx[:, i_idx].ravel()
    row_tgt = idx[:, j_idx].ravel()

    # Column constraints: for each col c, connect idx[i, c] <-> idx[j, c]
    col_src = idx[i_idx, :].ravel()
    col_tgt = idx[j_idx, :].ravel()

    # Both directions for each edge
    line_src = np.concatenate([row_src, row_tgt, col_src, col_tgt])
    line_tgt = np.concatenate([row_tgt, row_src, col_tgt, col_src])
    line_edges = np.stack([line_src, line_tgt], axis=0)

    region_edge_list = []
    for reg in np.unique(region):
        nodes = idx[region == reg].ravel()
        k = len(nodes)
        if k > 1:
            # Pairwise combinations
            ri, rj = np.triu_indices(k, k=1)
            src = nodes[ri]
            tgt = nodes[rj]
            # Both directions
            region_edge_list.append(np.stack([
                np.concatenate([src, tgt]),
                np.concatenate([tgt, src])
            ], axis=0))

    if region_edge_list:
        region_edges = np.concatenate(region_edge_list, axis=1)
    else:
        region_edges = np.empty((2, 0), dtype=np.int64)

    hetero_edge_index = {
        'line_constraint': torch.from_numpy(line_edges).contiguous(),
        'region_constraint': torch.from_numpy(region_edges).contiguous(),
        'diagonal_constraint': torch.from_numpy(diagonal_edges).contiguous(),
    }

    return hetero_edge_index


def build_heterogeneous_edge_index_cached(region: np.ndarray) -> Dict[str, torch.Tensor]:
    """Cached version of build_heterogeneous_edge_index.

    Computes edge indices only once per unique region layout.
    """
    key = _region_hash(region)

    if key not in _edge_index_cache:
        _edge_index_cache[key] = build_heterogeneous_edge_index(region)

    return _edge_index_cache[key]


def clear_edge_index_cache() -> int:
    """Clear the edge index cache and return number of entries removed."""
    count = len(_edge_index_cache)
    _edge_index_cache.clear()
    return count


def _canonical_img_key(src: str) -> str:
    """Extract base image filename (e.g., IMG_5193.jpg_rot0 -> IMG_5193.jpg)."""
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
    groups: Dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(_canonical_img_key(rec["source"]), []).append(rec)

    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val = int(len(keys) * val_ratio)
    val_keys = set(keys[:n_val])

    train, val = [], []
    for k, recs in groups.items():
        (val if k in val_keys else train).extend(recs)

    return train, val

class QueensDataset(Dataset):
    """PyTorch Geometric Dataset for Queens puzzle with heterogeneous edges and one-hot encoded regions."""

    # Cache of (train, val) splits so multiple Dataset instances reuse work
    _cache: Dict[tuple[Path, float, int], Tuple[list[dict], list[dict]]] = {}

    def __init__(
        self,
        json_path: str | Path,
        *,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")

        super().__init__(None, transform, pre_transform)

        self.json_path   = Path(json_path).expanduser()
        self.val_ratio   = val_ratio
        self.seed        = seed

        key = (self.json_path, val_ratio, seed)
        if key not in self._cache:
            records          = json.loads(self.json_path.read_text())
            train, val       = _split_by_img(records, val_ratio, seed)
            self._cache[key] = (train, val)

        train_set, val_set = self._cache[key]

        self.records = (
            train_set if split == "train"
            else val_set if split == "val"
            else train_set + val_set
        )

        self.max_regions = 11

    def len(self) -> int:
        return len(self.records)

    def get(self, idx: int) -> HeteroData:
        e = self.records[idx]

        region  = np.asarray(e["region"],        dtype=np.int64)   # (n, n)
        partial = np.asarray(e["partial_board"], dtype=np.int64)   # (n, n)
        label   = np.asarray(e["label_board"],   dtype=np.int64)   # (n, n)
        n       = region.shape[0]
        N2      = n * n

        coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)  # (N², 2)

        reg_onehot = np.zeros((N2, self.max_regions), dtype=np.float32)
        flat_ids   = region.flatten()
        reg_onehot[np.arange(N2), flat_ids] = 1.0                                  # (N², R)

        has_q = partial.flatten()[:, None].astype(np.float32)                      # (N², 1)

        x = torch.from_numpy(np.hstack([coords, reg_onehot, has_q]))               # (N², 3+R)

        y = torch.from_numpy(label.flatten().astype(np.int64))                     # (N²,)

        hetero_edge_index = build_heterogeneous_edge_index_cached(region)

        data = HeteroData()

        data['cell'].x = x
        data['cell'].y = y

        for edge_type, edge_index in hetero_edge_index.items():
            data['cell', edge_type, 'cell'].edge_index = edge_index

        data.n = torch.tensor([n], dtype=torch.long)
        data.step = torch.tensor([e["step"]], dtype=torch.long)
        data.meta = dict(source=e["source"], iteration=e["iteration"])

        return data

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
    """Return (train_loader, val_loader) with heterogeneous edge support."""
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

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        follow_batch=follow_batch or [],
    )

    train_loader = DataLoader(ds_train, shuffle=shuffle_train, **kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False,        **kwargs)

    return train_loader, val_loader

class MixedDataset(Dataset):
    """Dataset that mixes two datasets with stochastic sampling, ensuring all examples are used before reuse."""

    def __init__(self, dataset1, dataset2, ratio1=0.75):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.ratio1 = ratio1

        self.remaining_pool1 = list(range(len(dataset1)))
        self.remaining_pool2 = list(range(len(dataset2)))
        self._shuffle_pools()

        self._len = min(len(dataset1), len(dataset2))

        print(f"MixedDataset created:")
        print(f"  Dataset1 (state-0): {len(dataset1)} samples ({ratio1:.1%} sampling)")
        print(f"  Dataset2 (old filtered): {len(dataset2)} samples ({1-ratio1:.1%} sampling)")
        print(f"  Epoch length: {self._len}")

    def _shuffle_pools(self):
        """Shuffle both remaining pools."""
        random.shuffle(self.remaining_pool1)
        random.shuffle(self.remaining_pool2)

    def _sample_from_pool1(self):
        """Sample from dataset1, refill pool if exhausted."""
        if not self.remaining_pool1:
            self.remaining_pool1 = list(range(len(self.dataset1)))
            random.shuffle(self.remaining_pool1)

        idx = self.remaining_pool1.pop()
        return self.dataset1[idx]

    def _sample_from_pool2(self):
        """Sample from dataset2, refill pool if exhausted."""
        if not self.remaining_pool2:
            self.remaining_pool2 = list(range(len(self.dataset2)))
            random.shuffle(self.remaining_pool2)

        idx = self.remaining_pool2.pop()
        return self.dataset2[idx]

    def __getitem__(self, idx):
        """Stochastically sample from either dataset based on ratio."""
        if random.random() < self.ratio1:
            return self._sample_from_pool1()
        else:
            return self._sample_from_pool2()

    def __len__(self):
        return self._len

def create_filtered_old_dataset(json_path, val_ratio, seed, split="train"):
    """Create dataset from old multi-state data with iteration != 0 filtered out."""
    full_dataset = QueensDataset(
        json_path,
        split="train",
        val_ratio=val_ratio,
        seed=seed
    )

    filtered_records = [
        record for record in full_dataset.records
        if record.get('iteration', 0) != 0
    ]

    print(f"Filtered old dataset: {len(full_dataset.records)} -> {len(filtered_records)} (removed iteration 0)")

    filtered_dataset = QueensDataset.__new__(QueensDataset)
    filtered_dataset.__dict__.update(full_dataset.__dict__)

    if split == "train":
        train_records, val_records = _split_by_img(filtered_records, val_ratio, seed)
        filtered_dataset.records = train_records
    elif split == "val":
        train_records, val_records = _split_by_img(filtered_records, val_ratio, seed)
        filtered_dataset.records = val_records
    else:
        filtered_dataset.records = filtered_records

    return filtered_dataset

def hetero_to_homogeneous(hetero_data: HeteroData) -> Data:
    """Convert HeteroData to homogeneous Data by combining all edge types.

    This allows training homogeneous models (GAT, GNN) on heterogeneous graph data.
    All edge types (line, region, diagonal constraints) are merged into a single edge_index.
    """
    x = hetero_data['cell'].x
    y = hetero_data['cell'].y

    # Combine all edge types into a single edge_index
    edge_indices = []
    for edge_type in [('cell', 'line_constraint', 'cell'),
                      ('cell', 'region_constraint', 'cell'),
                      ('cell', 'diagonal_constraint', 'cell')]:
        if edge_type in hetero_data.edge_index_dict:
            edge_idx = hetero_data[edge_type].edge_index
            if edge_idx.numel() > 0:
                edge_indices.append(edge_idx)

    # Concatenate all edges
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create homogeneous Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # Preserve metadata
    if hasattr(hetero_data, 'n'):
        data.n = hetero_data.n
    if hasattr(hetero_data, 'step'):
        data.step = hetero_data.step
    if hasattr(hetero_data, 'meta'):
        data.meta = hetero_data.meta

    return data

class HomogeneousQueensDataset(QueensDataset):
    """Wrapper around QueensDataset that converts HeteroData to homogeneous Data.

    Use this for training GAT or GNN models that expect (x, edge_index) format.
    """

    def get(self, idx: int) -> Data:
        """Get item and convert to homogeneous format."""
        hetero_data = super().get(idx)
        return hetero_to_homogeneous(hetero_data)

def get_homogeneous_loaders(
    json_path: str,
    *,
    batch_size: int = 512,
    val_ratio: float = 0.10,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
):
    """Return (train_loader, val_loader) with homogeneous graphs for GAT/GNN models."""
    ds_train = HomogeneousQueensDataset(
        json_path,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
    )
    ds_val = HomogeneousQueensDataset(
        json_path,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(ds_train, shuffle=shuffle_train, **kwargs)
    val_loader = DataLoader(ds_val, shuffle=False, **kwargs)

    return train_loader, val_loader

def filter_dataset_to_step0(json_path: str | Path) -> List[dict]:
    """Load dataset and filter to only step=0 puzzles.

    Used for full-solve validation using StateValSet.json.
    Returns list of puzzle records.
    """
    json_path = Path(json_path).expanduser()
    records = json.loads(json_path.read_text())

    step0_records = [
        rec for rec in records
        if rec.get('step', 0) == 0
    ]

    print(f"Filtered {json_path.name}: {len(records)} total → {len(step0_records)} step-0 puzzles")
    return step0_records

def create_filtered_old_dataset_homogeneous(json_path, val_ratio, seed, split="train"):
    """Create homogeneous dataset from old multi-state data with iteration != 0 filtered out.

    This is the homogeneous version of create_filtered_old_dataset, returning Data instead of HeteroData.
    """
    full_dataset = HomogeneousQueensDataset(
        json_path,
        split="all",
        val_ratio=val_ratio,
        seed=seed
    )

    # The underlying records are the same, just need to filter
    filtered_records = [
        record for record in full_dataset.records
        if record.get('iteration', 0) != 0
    ]

    print(f"Filtered old dataset (homogeneous): {len(full_dataset.records)} -> {len(filtered_records)} (removed iteration 0)")

    # Create new dataset with filtered records
    filtered_dataset = HomogeneousQueensDataset.__new__(HomogeneousQueensDataset)
    filtered_dataset.__dict__.update(full_dataset.__dict__)

    if split == "train":
        train_records, val_records = _split_by_img(filtered_records, val_ratio, seed)
        filtered_dataset.records = train_records
    elif split == "val":
        train_records, val_records = _split_by_img(filtered_records, val_ratio, seed)
        filtered_dataset.records = val_records
    else:
        filtered_dataset.records = filtered_records

    return filtered_dataset

class BenchmarkDataset(vanillaDataset):
    """Secondary dataset for a non graph based model to be used for benchmarking."""
    def __init__(
        self,
        json_path: str | Path,
        *,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")

        super().__init__()

        self.json_path = Path(json_path).expanduser()
        self.val_ratio = val_ratio
        self.seed = seed

        records = json.loads(self.json_path.read_text())
        train_records, val_records = _split_by_img(records, val_ratio, seed)

        self.records = (
            train_records if split == "train"
            else val_records if split == "val"
            else train_records + val_records
        )

        self.max_regions = 11

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        e = self.record[idx]

        region  = np.asarray(e["region"],        dtype=np.int64)   # (n, n)
        partial = np.asarray(e["partial_board"], dtype=np.int64)   # (n, n)
        label   = np.asarray(e["label_board"],   dtype=np.int64)   # (n, n)
        n       = region.shape[0]
        N2      = n * n 

        #padding region to max size
        region_padded = self.pad(region, target_size=11, pad_with=-1)
        partial_padded = self.pad(partial, target_size=11, pad_with=0)
        label_padded = self.pad(label, target_size=11, pad_with=-100)

        coords = np.indices((self.max_regions, self.max_regions)).reshape(2, -1).T.astype(np.float32) / (self.max_regions - 1)  # (N², 2)
        reg_onehot = np.zeros((self.max_regions * self.max_regions, self.max_regions), dtype=np.float32)
        flat_ids   = region_padded.flatten()
        valid_mask = flat_ids != -1
        reg_onehot[np.arange(self.max_regions * self.max_regions)[valid_mask], flat_ids[valid_mask]] = 1.0                                  # (N², R)

        has_q = partial_padded.flatten()[:, None].astype(np.float32) # (N², 1)
        
        x = np.hstack([coords, reg_onehot, has_q]) # (N², 2+R+1)
        y = label_padded.flatten().astype(np.int64) # (N²,)

        sample = {
            "x": torch.from_numpy(x),               # (N², 2+R+1)
            "y": torch.from_numpy(y),               # (N²,)
            "n": n,
            "meta": dict(source=e["source"], iteration=e["iteration"])
        }
        return sample
 
    def pad(self, board: np.ndarray, target_size: int, pad_with: int) -> np.ndarray:
        """Pad the board to the target size with -1."""
        n = board.shape[0]
        if n >= target_size:
            return board
        padded_board = pad_with * np.ones((target_size, target_size), dtype=board.dtype)
        padded_board[:n, :n] = board
        return padded_board
    
def get_benchmark_loaders(
    json_path: str,
    *,
    batch_size: int = 512,
    val_ratio: float = 0.10,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
):
    """Return (train_loader, val_loader) for benchmark dataset."""
    ds_train = BenchmarkDataset(
        json_path,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
    )
    ds_val = BenchmarkDataset(
        json_path,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = vanillaDataLoader(ds_train, shuffle=shuffle_train, **kwargs)
    val_loader = vanillaDataLoader(ds_val, shuffle=False, **kwargs)

    return train_loader, val_loader