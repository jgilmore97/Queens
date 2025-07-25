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

def _build_edge_index(region: np.ndarray) -> torch.Tensor:
    """Return undirected edge_index tensor capturing Queens constraints."""
    n = region.shape[0]
    idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
    edges: list[Tuple[int, int]] = []

    # rows
    for r in range(n):
        for i, j in combinations(idx[r, :], 2):
            edges += [(i, j), (j, i)]

    # columns
    for c in range(n):
        for i, j in combinations(idx[:, c], 2):
            edges += [(i, j), (j, i)]

    # regions
    for reg in np.unique(region):
        nodes = idx[region == reg].ravel()
        for i, j in combinations(nodes, 2):
            edges += [(i, j), (j, i)]

    # immediate diagonals
    for r in range(n - 1):
        for c in range(n - 1):
            a, b = idx[r, c], idx[r + 1, c + 1]  # ↘
            edges += [(a, b), (b, a)]
        for c in range(1, n):
            a, b = idx[r, c], idx[r + 1, c - 1]  # ↙
            edges += [(a, b), (b, a)]

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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
        cache_edges: bool = True,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")

        super().__init__(None, transform, pre_transform)

        self.json_path   = Path(json_path).expanduser()
        self.val_ratio   = val_ratio
        self.seed        = seed
        self.cache_edges = cache_edges
        self._edge_cache: dict[int, torch.Tensor] = {}

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
        if board_key is not None and board_key in self._edge_cache:
            edge_index = self._edge_cache[board_key]
        else:
            edge_index = _build_edge_index(region)
            if board_key is not None:
                self._edge_cache[board_key] = edge_index

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
