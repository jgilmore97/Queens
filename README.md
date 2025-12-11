# Queens Puzzle ML Solver

A machine learning approach to solving Queens puzzle games using Graph Neural Networks with heterogeneous constraint modeling.

## Project Overview

Queens is a logic puzzle where players place n queens on an n×n colored board following these constraints:

- One queen per row and column
- One queen per color region
- No queens can touch diagonally (immediate adjacent diagonals only)

This project trains a **Hierarchical Reasoning Model (HRM)** to predict optimal queen placements autoregressively without backtracking. The model operates through two complementary mechanisms: a fast local constraint reasoner (L-module) and a slower global context builder (H-module) that iteratively refine predictions across multiple cycles.

## Architecture

### Hierarchical Reasoning Model (HRM)

The final architecture combines local and global reasoning in a structured hierarchy:

**L-Module (Local Constraint Reasoner)**
- Recurrent block with weight-tied layers: GAT → GAT → HGT
- Runs 2 micro-steps per reasoning cycle for convergence on immediate constraints
- Processes: line_constraint (row/col), region_constraint (color), diagonal_constraint (adjacency)
- Fixed architecture enables efficient iterative refinement

**H-Module (Global Context Manager)**
- Multi-head attention pooling over all node embeddings
- Produces global context vector z_H after L-module convergence
- Runs once per cycle (3 total cycles)

**Integration via FiLM Conditioning**
- Global context z_H modulates L-module activations across cycles
- Enables hierarchical convergence: local constraint detection → global consistency → solution refinement

**Readout Layer**
- Concatenates per-node features with global context
- MLP projects to per-cell logits
- Autoregressive inference: predicts next queen position, adds queen, repeats until n queens placed

### Graph Representation

Nodes: one per board cell
- Normalized row/column coordinates
- One-hot region ID (padded to max regions)
- Binary has-queen flag

Heterogeneous Edges: enables differentiated attention learning
- line_constraint: cells in same row or column
- region_constraint: cells in same color region
- diagonal_constraint: cells at immediate diagonal adjacency

This explicit constraint encoding allows the model to learn specialized attention patterns for each constraint type rather than inferring constraints from raw spatial features.

## Performance

### Single State Validation Metrics - Single State means evaluated on ability to make a single correct placement based on being given a perfectly filled out board up to that point. IE can model place Queen 5 correctly given it has a board with queens 1-4 placed correctly as input. This Val set does include empty boards as well.
- F1 Score: 99.36%
- Top-1 Accuracy: 99.99999%

###  Full Solve Val Results (720 Unseen Official Linkedin Puzzles) - Full Solve refers to the puzzles ability to auto-regressively solve an entire puzzle perfectly. IE if there are 8 queens to be placed, can we auto-regressively run the model 8 times to solve the puzzle with no errors. This set includes 180 official linkedin puzzles that are also augmented via rotation so every puzzle becomes 4 rotations.
- Perfect-Solve Rate (First-Try): 98.8%
- Inference Time: ~0.5s per puzzle (CPU)
- Failure Mode: Errors concentrated in early steps (0-2), indicating ambiguity in initial placements
- Behavior: Model either solves completely or fails irrecoverably; no partial/recoverable errors

###  Full Solve Test Results (70 More Unseen Official Linkedin Puzzles)
- Perfect-Solve Rate (First-Try): 100%

### Key Characteristics
- No backtracking required; all placements are direct predictions
- Model failure indicates insufficient reasoning depth for that puzzle instance
- Early-step failures suggest inherent ambiguity in initial board configuration

## Training Setup

Loss: Binary focal loss (α=0.25, γ=2.0) for handling class imbalance
Optimizer: AdamW (lr=1e-3, wd=1e-5)
Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
Batch Size: 512
Epochs: 18

Dataset Transition: Switch to state-0 (empty board) dataset at epoch 5 to improve early-step accuracy and reduce Type-2 errors.

## Data & Labeling

### Dataset Generation
1. Create 10k base puzzles through region boundary mutation (ensures single-solution constraint)
2. Augment with 4-way rotations (40k total)
3. Generate progressive game states by iteratively removing queens (350k training examples)

### Move Type Classification
- Type 1: Immediately illegal (violates row/col/region/diagonal). Label: 0
- Type 2: Locally legal but globally invalid (leads to dead end). Label: 0
- Type 3: Part of valid solution path. Label: 1

The core challenge is distinguishing Type 2 from Type 3 using learned global reasoning. The hierarchical mechanism explicitly addresses this through multi-cycle refinement: initial L-module passes detect immediate constraints, while H-module integrates global state to identify unsolvable downstream positions.

## Project Structure

```
code
├── model.py                      # HRM and supporting architectures
├── data_loader.py               # PyTorch Geometric dataset + heterogeneous graph construction
├── config.py                    # Centralized configuration
├── train.py                     # Training loop, loss computation, metrics
├── board_manipulation.py        # Image processing, region mutation, synthetic generation
├── FullRun.py                   # Experiment orchestrator
├── evaluation_analysis.py       # Evaluation pipeline and visualization tools
├── evaluation_util.py           # Utility functions for puzzle evaluation
├── experiment_tracker.py  # Weights & Biases logging and checkpointing
├── solver.py                    # Traditional backtracking solver (reference/validation)
└── solver.py           # Enhanced solver with cycle detection and memoization
```

## Training Details

### State-0 Dataset Strategy
Mid-training transition to state-0 dataset (empty boards only) addresses the Type-2 vs Type-3 distinction more directly. Empty boards present the maximum ambiguity, forcing the model to develop robust global reasoning patterns that generalize to partial-board states.

### Loss Function & Class Imbalance
Binary focal loss with α=0.25 (weighting), γ=2.0 (focusing) aggressively downweights easy negatives and emphasizes hard positives, critical for navigating the sparse positive class in this problem.

### Heterogeneous Graph Modeling
Each edge type receives independent attention mechanisms, allowing the model to learn constraint-specific reasoning strategies rather than conflating different constraints into a single attention pattern.

## Visualization & Analysis

### Current Focus: Layer-Activation Reasoning Visualization

Objective: Demonstrate how the model reasons across hierarchical depth rather than only which cell it predicts.

Approach: Custom layer-activation visualization methods that display:
- Early reasoning (L-module, micro-step 1): Immediate constraint detection
- Mid reasoning (L-module, micro-step 2 + H-module update): Global context integration
- Late reasoning (final cycle): Solution refinement and convergence

Prior attempts (integrated gradients, attention maps) highlighted only correct predictions or captured too narrow attention focus. Target outcome: visualization revealing evolving rule awareness and constraint interaction patterns, demonstrating hierarchical convergence for portfolio presentation.

### Analysis Tools
- Board size effects: Performance scaling across 7×7 to 11×11 puzzles
- Game state analysis: Accuracy vs. queens remaining
- Spatial patterns: Error heatmaps by position (center/edge/corner)
- Early-step error distribution: Identify systematic early-placement ambiguities

## Configuration

Key parameters in `config.py`:
- Model: 2 GAT layers, 2 HGT layers, 128 hidden dimension
- Cycles: 3 (H-module updates per prediction)
- Micro-steps: 2 per cycle (L-module iterations)
- Input injection: Enabled at each cycle
- Dropout: 0.10
- Batch size: 512

## Experiment Tracking

Weights & Biases integration for reproducibility:
- Training/validation metrics (loss, F1, top-1 accuracy)
- Gradient analysis for vanishing gradient detection
- Prediction samples and confidence distributions
- System resource monitoring

## Learning Objectives

- Graph Neural Network design for constraint satisfaction problems
- Hierarchical reasoning architecture for multi-scale problem solving
- Heterogeneous graph modeling for differentiated constraint reasoning
- Autoregressive decoding without backtracking
- Systematic evaluation, visualization, and analysis of learned reasoning
- Experiment tracking and reproducibility at scale

### Why Hierarchical Reasoning?

Single-stage models struggle with Type-2 errors because they must balance two conflicting objectives simultaneously:
1. Detect immediate constraint violations (local reasoning)
2. Identify globally unsolvable positions (global reasoning)

The HRM separates these concerns:
- Fast local iterations converge on immediate constraints
- Slow global update builds context from accumulated local state
- Cycling enables progressive refinement without exponential cost

This structure maps to the problem's inherent difficulty hierarchy: early steps require global reasoning (most ambiguous), while later steps are dominated by local constraint elimination (highly determined).
