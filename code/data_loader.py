from __future__ import annotations

import json
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData, Dataset 
from torch_geometric.loader import DataLoader

def build_heterogeneous_edge_index(region: np.ndarray) -> Dict[str, torch.Tensor]:
    """Return dictionary of edge_index tensors for each constraint type."""
    n = region.shape[0]
    idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
    
    # Dictionary to store different edge types
    edge_dict = {
        'line_constraint': [],
        'region_constraint': [],
        'diagonal_constraint': []
    }

    # LINE CONSTRAINTS (rows and columns combined)
    # Rows
    for r in range(n):
        for i, j in combinations(idx[r, :], 2):
            edge_dict['line_constraint'].extend([(i, j), (j, i)])
    
    # Columns
    for c in range(n):
        for i, j in combinations(idx[:, c], 2):
            edge_dict['line_constraint'].extend([(i, j), (j, i)])

    # REGION CONSTRAINTS (same color)
    for reg in np.unique(region):
        nodes = idx[region == reg].ravel()
        for i, j in combinations(nodes, 2):
            edge_dict['region_constraint'].extend([(i, j), (j, i)])

    # DIAGONAL CONSTRAINTS
    for r in range(n - 1):
        for c in range(n - 1):
            # Down right
            a, b = idx[r, c], idx[r + 1, c + 1]
            edge_dict['diagonal_constraint'].extend([(a, b), (b, a)])
        for c in range(1, n):
            # Down left
            a, b = idx[r, c], idx[r + 1, c - 1]
            edge_dict['diagonal_constraint'].extend([(a, b), (b, a)])

    # Convert to tensors
    hetero_edge_index = {}
    for edge_type, edges in edge_dict.items():
        if edges: 
            hetero_edge_index[edge_type] = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            hetero_edge_index[edge_type] = torch.empty((2, 0), dtype=torch.long)

    return hetero_edge_index


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
    PyTorch Geometric Dataset for the Queens puzzle with heterogeneous edges.
    - Region IDs are one-hot encoded and padded to the largest board in the JSON.
    - Row/col coordinates are min-max scaled to the [0 , 1] range.
    - Returns HeteroData objects with separate edge types for different constraints.
    """

    # cache of (train, val) splits so multiple Dataset instances reuse work
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

        # ------------------------------------------------------
        # 1) build / fetch cached train-val split
        # ------------------------------------------------------
        key = (self.json_path, val_ratio, seed)
        if key not in self._cache:
            records          = json.loads(self.json_path.read_text())
            train, val       = _split_by_img(records, val_ratio, seed)
            self._cache[key] = (train, val)

        train_set, val_set = self._cache[key]

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
        self.max_regions = 11
        # max(
        #     int(np.max(r["region"])) + 1
        #     for r in (train_set + val_set)
        # )

    # PyG Dataset hooks
    def len(self) -> int:  # noqa: D401
        return len(self.records)

    def get(self, idx: int) -> HeteroData:
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

        # --- heterogeneous edge_index ------------------------------------
        hetero_edge_index = build_heterogeneous_edge_index(region)

        # --- assemble HeteroData object ----------------------------------
        data = HeteroData()
        
        # Add node information (all nodes are the same type - 'cell')
        data['cell'].x = x
        data['cell'].y = y
        
        # Add edge information for each constraint type
        for edge_type, edge_index in hetero_edge_index.items():
            data['cell', edge_type, 'cell'].edge_index = edge_index
        
        # Add metadata
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
    """
    Return (train_loader, val_loader) with heterogeneous edge support.

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

class MixedDataset(Dataset):
    """
    Dataset that mixes two datasets with stochastic sampling while ensuring 
    all examples are used before reuse.
    """
    def __init__(self, dataset1, dataset2, ratio1=0.75):
        """
        Args:
            dataset1: First dataset (typically state-0, gets ratio1 probability)
            dataset2: Second dataset (typically filtered old dataset, gets 1-ratio1 probability)  
            ratio1: Probability of sampling from dataset1 (0.75 = 75%)
        """
        self.dataset1 = dataset1  # state-0 dataset
        self.dataset2 = dataset2  # filtered old dataset
        self.ratio1 = ratio1
        
        # Remaining pools - refilled when exhausted
        self.remaining_pool1 = list(range(len(dataset1)))
        self.remaining_pool2 = list(range(len(dataset2)))
        self._shuffle_pools()
        
        # Length is minimum of the two datasets
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
    from data_loader import QueensDataset
    
    # Create dataset but we'll filter it
    full_dataset = QueensDataset(
        json_path,
        split="all",  # Get all data first
        val_ratio=val_ratio,
        seed=seed
    )
    
    # Filter out iteration 0 examples
    filtered_records = [
        record for record in full_dataset.records 
        if record.get('iteration', 0) != 0
    ]
    
    print(f"Filtered old dataset: {len(full_dataset.records)} -> {len(filtered_records)} (removed iteration 0)")
    
    # Create new dataset with filtered records
    filtered_dataset = QueensDataset.__new__(QueensDataset)
    filtered_dataset.__dict__.update(full_dataset.__dict__)
    
    # Apply train/val split to filtered records  
    if split == "train":
        # Use same splitting logic as original QueensDataset
        from data_loader import _split_by_img
        train_records, val_records = _split_by_img(filtered_records, val_ratio, seed)
        filtered_dataset.records = train_records
    elif split == "val":
        from data_loader import _split_by_img
        train_records, val_records = _split_by_img(filtered_records, val_ratio, seed)
        filtered_dataset.records = val_records
    else:
        filtered_dataset.records = filtered_records
    
    return filtered_dataset