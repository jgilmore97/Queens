# Queens Puzzle ML Solver

A machine learning approach to solving the LinkedIn Queens puzzle using Graph Neural Networks with hierarchical constraint reasoning. The final model solves 100% of a held-out test set of 128 official LinkedIn puzzles and 99.9% of a 716-puzzle validation set — with no backtracking.

> For a full breakdown of the architecture, training process, and results, visit **[jgilmore97.github.io/Queens](https://jgilmore97.github.io/Queens/)** or read [writeup.md](writeup.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/jgilmore97/Queens.git
cd Queens

# Install in editable mode
pip install -e .

# Or with all optional dependencies (dev tools, gradio app, hyperparameter sweeps)
pip install -e ".[all]"
```

## Data Setup

Training data is not included due to size. Place the following files in the `data/` directory:

| File | Description |
|------|-------------|
| `StateTrainingSet.json` | Training data with progressive game states (~350k examples) |
| `State0TrainingSet.json` | State-0 (empty board) training examples |
| `StateValSet.json` | Validation set for full-solve evaluation |

## Quick Start

```bash
# Train the HRM model
make train

# Run ablation study
make ablation

# Launch the interactive web demo
make app
```

## Results

**Solver comparison** — HRM vs. classical solvers on 128 official LinkedIn test puzzles:

| Solver | Solve Rate | Avg Time | Avg Guesses | Avg Failed Guesses |
|--------|-----------|----------|-------------|-------------------|
| Backtracking | 100% | 3.80 ms | 997 | 988 |
| AC-3 | 100% | 14.13 ms | 454 | 446 |
| OR-Tools CP-SAT | 100% | 6.24 ms | 13 | 0.7 |
| Neural (HRM) | 100% | 87.99 ms | — | — |

The neural model makes zero failed guesses because it never backtracks — every queen placement is final.

**Ablation** — architectural progression under controlled conditions:

| Model | Parameters | Single-State F1 | Validation Set Full Solve Rate |
|-------|------------|-----------------|--------------------------------|
| GAT | 86K | 76.6% | 45.3% |
| HeteroGAT | 445K | 96.0% | 91.0% |
| Ablation HRM | 359K | 99.5% | 97.9% |
| Benchmark HRM | 446K | 92.9% | 81.5% |
| Benchmark Sequential | 1.2M | 91.4% | 82.2% |
| Final HRM | 359K | 99.5% | 99.9% |
