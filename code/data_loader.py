from __future__ import annotations

import json
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader


class EdgeIndexCache:
    """Cache for row/col/diag edges by board size."""
    def __init__(self):
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n not in self._cache:
            self._cache[n] = self._compute(n)
        return self._cache[n]

    def _compute(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
        # Rows
        rows = []
        for r in range(n):
            i, j = np.triu_indices(n, k=1)
            rows.append(np.stack([idx[r, i], idx[r, j]], axis=1))
        row_edges = np.vstack(rows)
        # Cols
        cols = []
        for c in range(n):
            i, j = np.triu_indices(n, k=1)
            cols.append(np.stack([idx[i, c], idx[j, c]], axis=1))
        col_edges = np.vstack(cols)
        # Diagonals
        diag1 = np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1)
        diag2 = np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
        diag_edges = np.vstack([diag1, diag2])
        return row_edges, col_edges, diag_edges


def _compute_region_edges(region: np.ndarray) -> np.ndarray:
    flat = region.ravel()
    uniq, inv = np.unique(flat, return_inverse=True)
    edges = []
    for rid in range(len(uniq)):
        cells = np.where(inv == rid)[0]
        if len(cells) > 1:
            i, j = np.triu_indices(len(cells), k=1)
            edges.append(np.stack([cells[i], cells[j]], axis=1))
    return np.vstack(edges) if edges else np.zeros((0, 2), dtype=np.int64)


def _build_edge_index(region: np.ndarray, cache: EdgeIndexCache) -> torch.Tensor:
    n = region.shape[0]
    row, col, diag = cache.get(n)
    reg = _compute_region_edges(region)
    all_edges = np.vstack([row, col, diag, reg])
    und = np.vstack([all_edges, all_edges[:, [1, 0]]])
    return torch.tensor(und.T, dtype=torch.long)


def _canonical_img_key(src: str) -> str:
    m = re.match(r"^(.*?\.jpg)", src, re.IGNORECASE)
    if not m:
        raise ValueError(f"Bad source='{src}'")
    return m.group(1)


def _split_by_img(records: List[dict], val_ratio: float, seed: int) -> tuple[List[dict], List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in records:
        key = _canonical_img_key(r["source"])
        groups.setdefault(key, []).append(r)
    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = int(len(keys) * val_ratio)
    val_keys = set(keys[:n_val])
    train, val = [], []
    for k, recs in groups.items():
        (val if k in val_keys else train).extend(recs)
    return train, val


class QueensDataset(InMemoryDataset):
    """In-memory Queens dataset with pre-built graphs and feature caching."""
    def __init__(
        self,
        json_path: str | Path,
        split: str = "all",
        val_ratio: float = 0.2,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.json_path = Path(json_path).expanduser()
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.cache = EdgeIndexCache()
        root = str(self.json_path.parent)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [self.json_path.name]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"data_{self.split}.pt"]

    def download(self):
        pass

    def process(self):
        records = json.loads(self.json_path.read_text())
        train, val = _split_by_img(records, self.val_ratio, self.seed)
        if self.split == "train":
            subset = train
        elif self.split == "val":
            subset = val
        else:
            subset = train + val

        max_regions = max(int(np.max(r["region"])) + 1 for r in subset)
        data_list = []
        for rec in subset:
            region = np.asarray(rec["region"], dtype=np.int64)
            partial = np.asarray(rec["partial_board"], dtype=np.int64)
            label = np.asarray(rec["label_board"], dtype=np.int64)
            n = region.shape[0]
            N2 = n * n
            coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)
            onehot = np.zeros((N2, max_regions), dtype=np.float32)
            ids = region.ravel()
            onehot[np.arange(N2), ids] = 1.0
            hasq = partial.ravel()[:, None].astype(np.float32)
            x = torch.from_numpy(np.hstack([coords, onehot, hasq]))
            y = torch.from_numpy(label.ravel().astype(np.int64))
            ei = _build_edge_index(region, self.cache)
            data_list.append(Data(x=x, edge_index=ei, y=y, n=torch.tensor([n]), step=torch.tensor([rec.get("step",0)]), meta=dict(source=rec.get("source"), iteration=rec.get("iteration"))))

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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
) -> tuple[DataLoader, DataLoader]:
    ds_train = QueensDataset(json_path, split="train", val_ratio=val_ratio, seed=seed)
    ds_val = QueensDataset(json_path, split="val",   val_ratio=val_ratio, seed=seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, follow_batch=follow_batch or [])
    train_loader = DataLoader(ds_train, shuffle=shuffle_train, **kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False,        **kwargs)
    return train_loader, val_loader
