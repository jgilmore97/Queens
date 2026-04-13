"""
FastAPI server — thin wrapper around the existing Solver for browser extension use.

Usage:
    uvicorn server.app:app --reload

Weights are downloaded automatically from HuggingFace on first run and cached at
~/.cache/huggingface/hub/. Subsequent starts are instant.
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# Allow running from project root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from queens_solver.evaluation.solver import Solver 

HF_REPO_ID = "Jgilmore/QueensHierarchicalGNN"
HF_FILENAME = "HierarchicalGNN.pt"

EXTENSION_DATA_PATH = Path(__file__).parent.parent / "data" / "extension_solved_boards.json"

_solver: Solver | None = None


def _save_board(region: np.ndarray, queen_board: np.ndarray) -> bool:
    """Append a solved puzzle to extension_solved_boards.json.

    Skips duplicates by comparing region matrices.
    Returns True if saved, False if duplicate.
    """
    region_list = region.tolist()

    existing: list = []
    if EXTENSION_DATA_PATH.exists():
        with open(EXTENSION_DATA_PATH, "r") as f:
            existing = json.load(f)

    if any(entry["region"] == region_list for entry in existing):
        return False

    existing.append({
        "region": region_list,
        "label_board": queen_board.tolist(),
        "source": f"extension_{datetime.now().strftime('%Y-%m-%d')}",
    })

    EXTENSION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTENSION_DATA_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _solver
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fetching model weights from {HF_REPO_ID!r} (cached after first download)…")
    checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    print(f"Loading model on {device}…")
    _solver = Solver(model_path=checkpoint, device=device)
    print("Model ready.")
    yield


app = FastAPI(title="Queens Solver API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    region: List[List[int]]
    batch_placement: bool = True


class SolveStep(BaseModel):
    queens: List[List[int]]
    l_heatmap: Optional[List[List[float]]]
    h_activation_heatmap: Optional[List[List[float]]]
    h_heatmap: Optional[List[List[float]]]


class SolveResponse(BaseModel):
    steps: List[SolveStep]


def _heatmap_to_list(act_dict: dict, module: str, prefer: str = 'late') -> Optional[List[List[float]]]:
    """Extract a heatmap for a given module, preferring a specific cycle stage."""
    maps = act_dict.get(module, {})
    keys = [prefer] + [k for k in ('late', 'mid', 'early') if k != prefer]
    for key in keys:
        heatmap = maps.get(key)
        if heatmap is not None:
            return heatmap.tolist()
    return None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _solver is not None}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if _solver is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    region = np.array(req.region)
    queen_board, _, activations = _solver.solve_puzzle(
        {"region": region}, capture_activations=True, batch_placement=req.batch_placement
    )

    _save_board(region, queen_board)

    steps = []
    for act_dict in activations:
        steps.append(SolveStep(
            queens=[[r, c] for r, c in act_dict.get('queens', [])],
            l_heatmap=_heatmap_to_list(act_dict, 'L', prefer='late'),
            h_activation_heatmap=_heatmap_to_list(act_dict, 'H', prefer='late'),
            h_heatmap=_heatmap_to_list(act_dict, 'H_attention', prefer='late'),
        ))

    return SolveResponse(steps=steps)
