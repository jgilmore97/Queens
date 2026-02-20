# Queens Puzzle ML Solver

A machine learning approach to solving Queens puzzle games using Graph Neural Networks with hierarchical constraint reasoning.

## Project Overview

Queens is a logic puzzle where players place n queens on an n×n colored board. Each queen must be the only one in its row, the only one in its column, the only one in its color region, and cannot touch any other queen diagonally. The puzzle always has exactly one valid solution.

This project trains a Hierarchical Reasoning Model (HRM) to predict optimal queen placements autoregressively without backtracking. The model learns to reason about local constraints through graph attention and global board state through hierarchical context aggregation.

## Development Progression

The path to the final architecture followed an iterative process of identifying limitations and addressing them with increasingly sophisticated methods.

**Problem Scoping.** The core challenge in Queens is distinguishing between moves that are locally legal but globally invalid (leading to dead ends) versus moves that are part of a valid solution path. A cell might satisfy all immediate constraints (no queen in its row, column, region, or diagonal neighbors) while still being a wrong choice because it eliminates all valid placements for some future queen. Detecting this requires reasoning about the entire board state, not just local neighborhoods.

**Graph Representation.** The decision to represent the board as a graph came from recognizing that the constraints are relational. Row and column constraints connect cells linearly, region constraints connect irregularly shaped groups, and diagonal constraints connect adjacent corners. Rather than forcing a CNN to learn these relationships implicitly from spatial position, encoding them explicitly as edges lets the model learn specialized attention patterns for each constraint type. The graph has one node per cell with three edge types: line constraints (same row or column), region constraints (same color region), and diagonal constraints (immediate diagonal adjacency).

**GAT.** The first model used standard Graph Attention Networks over a homogeneous graph. This established the basic approach of learning attention-weighted message passing between constrained cells, but treated all constraints identically. The model achieved about 76% F1 on single-state prediction and solved only 45% of puzzles end-to-end. The failure pattern showed errors distributed throughout the solve sequence, suggesting the model lacked the representational capacity to distinguish constraint types.

**HeteroGAT.** The next iteration introduced heterogeneous graph convolutions with separate attention mechanisms for each edge type, plus HGT layers at intermediate depths for cross-constraint integration within each node's neighborhood. This allowed the model to learn different reasoning patterns for row/column constraints versus region constraints versus diagonal constraints. Performance improved substantially to 96% F1 and 91% full solve rate. However, errors still concentrated in early steps where global reasoning matters most.

**HRM.** The final architecture separates local and global reasoning into distinct modules that iterate in cycles. The L-module runs multiple micro-steps of graph attention to converge on local constraint detection. The H-module then aggregates global context through multi-head attention pooling. This cycle repeats, allowing the model to progressively refine its understanding from local constraints to global consistency to solution confidence. This structure directly addresses the core problem: early placements require mostly global reasoning (many cells satisfy local constraints), while later placements are dominated by local constraint elimination (few valid options remain).

**Benchmark Comparison.** To validate that the architectural choices matter, two benchmark models were trained on the same data. A Benchmark HRM uses the same hierarchical iteration pattern but replaces graph convolutions with standard transformer layers over a flattened board representation. A Benchmark Sequential uses a simple stacked transformer without hierarchical structure. Both benchmarks achieved around 82% full solve rate despite the Benchmark Sequential having over three times the parameters of the HRM. The Benchmark HRM's performance demonstrates that hierarchical reasoning helps, but the HRM's superior results show that graph structure provides additional benefits that transformers alone cannot match.

## Architecture

The HRM combines local constraint reasoning with global context in a structured hierarchy.

**L-Module (Local Constraint Reasoner).** A recurrent block with weight-tied layers processes the heterogeneous graph. Each micro-step applies two HeteroConv layers (one GAT per edge type) followed by an HGT layer for cross-constraint integration. The module runs 2 micro-steps per cycle to allow local constraint information to propagate and converge. The heterogeneous structure means the model can learn that row/column constraints should be processed differently than region constraints or diagonal constraints.

**H-Module (Global Context Manager).** After L-module convergence, multi-head attention pooling aggregates all node embeddings into a global context vector. This captures board-wide patterns that individual nodes cannot see through local message passing alone. The H-module runs once per cycle, operating at a slower timescale than the local reasoning.

**Cycle Integration.** The global context conditions the next cycle's L-module processing. Three cycles allow for progressive refinement: initial passes detect immediate constraint violations, middle passes integrate global state to identify problematic positions, and final passes converge on solution confidence.

**Readout.** The final node embeddings concatenated with global context pass through an MLP to produce per-cell logits. During inference, the model predicts the next queen position as the argmax, places that queen, updates the board state, and repeats until all queens are placed.

**Graph Representation.** Each cell becomes a node with features encoding normalized row/column coordinates, one-hot region ID, and a binary flag indicating whether a queen is already placed. Edges encode the three constraint types explicitly, enabling the model to learn specialized attention patterns rather than inferring constraints from spatial position alone.

## Performance

The model is trained on single-state prediction and evaluated on both single-state accuracy and full autoregressive solving.

**Training Objective (Single-State Prediction).** Given a board with some queens already correctly placed, predict which remaining cells are valid for the next queen. This is a multi-label classification problem where the model sees partial solutions and learns to identify legal next moves. The training set includes boards at all stages of completion, from empty boards to boards with only one queen remaining to place.

**Single-State Validation Metrics.** On held-out partial board states, the best model achieves 99.54% F1 and 99.98% Top-1 accuracy. Top-1 measures whether the model's highest-confidence prediction is a valid move, which is what matters for autoregressive solving.

**Full Autoregressive Solving (Validation).** The validation set contains 180 official LinkedIn puzzles augmented with 4-way rotations for 720 total puzzles. Full solving means starting from an empty board and running the model repeatedly to place all queens without any errors. The model solves 715 of 720 puzzles (99.3%) on the first attempt. When the model fails, it fails early (steps 0-2) where global ambiguity is highest, and failures are not recoverable since one wrong placement cascades into an unsolvable position.

**Full Autoregressive Solving (Test).** The test set contains 97 official LinkedIn puzzles collected separately from the validation set. LinkedIn releases one puzzle per day, so the test set grows over time. The model solves all 97 test puzzles (100%) on the first attempt.

**Inference Speed.** About 0.5 seconds per puzzle on CPU.

## Ablation Study

An ablation study compared the architectural progression under controlled conditions. All models were trained on the same data with comparable hyperparameter budgets. These results informed the decision to focus development effort on the HRM architecture rather than representing final tuned performance.

| Model | Parameters | Single-State F1 | Full Solve Rate |
|-------|------------|-----------------|-----------------|
| GAT | 86K | 76.6% | 45.3% |
| HeteroGAT | 445K | 96.0% | 91.0% |
| HRM | 359K | 99.5% | 97.9% |
| Benchmark HRM | 446K | 92.9% | 81.5% |
| Benchmark Sequential | 1.2M | 91.4% | 82.2% |

The progression from GAT to HeteroGAT shows the value of constraint-specific attention. The jump from HeteroGAT to HRM shows the value of hierarchical local-global iteration. The benchmark comparison shows that the HRM's graph structure contributes meaningfully beyond what hierarchical reasoning alone provides. The Benchmark Sequential's lower performance despite having over three times the parameters suggests that parameter count is not the limiting factor.

## Data & Labeling

The training dataset starts with 10k base puzzles generated through region boundary mutation, a process that guarantees each puzzle has exactly one valid solution. These are augmented with 4-way rotations to produce 40k puzzles, then expanded into progressive game states by iteratively removing queens from solved boards. This yields about 350k training examples representing boards at various stages of completion.

Each cell in a training example is labeled as valid (1) or invalid (0) for the next queen placement. Invalid cells fall into two categories: those that immediately violate a constraint (same row, column, region, or diagonal as an existing queen) and those that are locally legal but globally invalid because they lead to unsolvable positions. The model must learn to distinguish these cases, which is the core difficulty of the problem. Type-1 violations are easy to detect through local reasoning, but Type-2 violations require understanding the global board state to recognize that a placement eliminates all valid options for some future queen.

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

The training data is not included in this repository due to size. Place the following JSON files in the `data/` directory:

| File | Description |
|------|-------------|
| `StateTrainingSet.json` | Training data with progressive game states (350k examples) |
| `State0TrainingSet.json` | State-0 (empty board) training data for early-step accuracy |
| `StateValSet.json` | Validation set for autoregressive full-solve evaluation |

The expected directory structure:
```
data/
├── StateTrainingSet.json
├── State0TrainingSet.json
└── StateValSet.json
```

## Quick Start

```bash
# Train the HRM model
make train
# or: python scripts/train.py

# Run ablation study
make ablation

# Launch the interactive web demo
make app
```

## Project Structure

```
queens-solver/
├── data/                           # Dataset files (not tracked)
│   ├── StateTrainingSet.json
│   ├── State0TrainingSet.json
│   └── StateValSet.json
├── src/
│   └── queens_solver/              # Main package
│       ├── __init__.py
│       ├── config.py               # Centralized configuration
│       ├── models/                 # Model architectures
│       │   ├── models.py           # HRM, HeteroGAT, GAT implementations
│       │   └── benchmark.py        # Benchmark models for comparison
│       ├── data/                   # Data loading & processing
│       │   ├── dataset.py          # PyTorch Geometric datasets
│       │   ├── preprocessing.py    # Image processing, puzzle generation
│       │   └── utils.py            # Data utilities
│       ├── training/               # Training logic
│       │   ├── trainer.py          # Training loops and metrics
│       │   ├── benchmark_trainer.py # Benchmark model training
│       │   └── tracker.py          # W&B experiment tracking
│       └── evaluation/             # Evaluation & inference
│           ├── solver.py           # Model-based puzzle solver
│           ├── evaluator.py        # Full puzzle evaluation
│           ├── benchmark_eval.py   # Benchmark evaluation
│           └── utils.py            # Evaluation utilities
├── scripts/                        # Entry points
│   ├── train.py                    # Main training script
│   ├── ablation.py                 # Ablation study runner
│   └── sweep.py                    # Hyperparameter sweep
├── app/                            # Web interface
│   └── gradio_app.py               # Gradio demo application
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Dependencies
├── Makefile                        # Common commands
└── README.md
```

## Training Details

Training uses AdamW optimizer (lr=1e-3, weight decay=1e-5) with ReduceLROnPlateau scheduling (patience=5, factor=0.5) for 18 epochs at batch size 512. The loss function is binary focal loss (α=0.25, γ=2.0) to handle severe class imbalance where most cells are invalid placements. Focal loss downweights easy negatives and emphasizes hard positives, which is critical when the model needs to learn subtle distinctions between locally-legal-but-globally-invalid moves.

Mid-training at epoch 5, the dataset transitions to include more state-0 (empty board) examples. Empty boards present maximum ambiguity since many cells satisfy local constraints, forcing the model to develop robust global reasoning patterns. This strategy directly targets the hardest cases where errors are most likely.

## Visualization

The model saves intermediate activations during forward passes to enable visualization of its reasoning process. By computing norms of the activations at each layer and cycle, we can see how the model responds to existing queens and applies game constraints. This is not a perfect reflection of internal representations but provides an intuitive picture of how constraint awareness develops across the hierarchical processing stages. Early cycles show the model detecting local constraint violations, while later cycles show convergence toward the predicted placement.

## Configuration

The default configuration in `config.py` uses 128 hidden dimensions, 3 hierarchical cycles with 2 micro-steps each, and 0.10 dropout. All hyperparameters are centralized there for easy modification during experimentation.

## Experiment Tracking

Weights & Biases integration logs training and validation metrics (loss, F1, top-1 accuracy), gradient statistics for debugging vanishing gradients, prediction samples with confidence distributions, and system resource usage.

## Learning Objectives

This project explores Graph Neural Network design for constraint satisfaction problems, hierarchical reasoning architectures that separate local and global processing, heterogeneous graph modeling for constraint-specific attention, and autoregressive decoding without backtracking. The experimental progression demonstrates systematic ablation methodology and the importance of architectural choices over raw parameter count.
