# Queens Puzzle ML Solver - WIP

A machine learning approach to solving Queens puzzle games using Graph Neural Networks with heterogeneous constraint modeling.

## 🎯 Project Overview

Queens is a logic puzzle where players place n queens on an n×n colored board following these constraints:
- One queen per row and column
- One queen per color region
- No queens can touch diagonally

This project trains a **Hybrid GAT-HGT model** to predict optimal queen placements and integrates it with a traditional backtracking solver for robust puzzle solving.

## 🏗️ Architecture

### Core Components
- **Hybrid GAT-HGT Model**: Combines constraint-specific attention with global context
  - `line_constraint`: Row/column mutual exclusion
  - `region_constraint`: Color region mutual exclusion  
  - `diagonal_constraint`: Diagonal adjacency prevention

- **Enhanced Solver**: ML-guided backtracking with smart path prioritization
  - Cycle detection and failed move memoization
  - Dynamic top-k candidate selection
  - Multi-level backtracking for unsolvable state recovery

### Data Pipeline
1. **Image Processing**: Convert puzzle photos to integer-labeled matrices
2. **Data Generation**: Mutate source puzzles by growing/shrinking regions (10k puzzles)
3. **Augmentation**: 4-way rotational augmentation (40k total)
4. **State Sequences**: Generate progressive game states (350k training examples)

## 📊 Performance

- **Training Data**: 350k examples from 10k base puzzles
- **Validation Accuracy**: 99.9% top-1 accuracy
- **Model Size**: 3-layer Hybrid GAT-HGT with ~85k parameters
- **Solver Integration**: ML guidance with traditional backtracking fallback

# Project Structure
├── board_manipulation.py    # Image processing & data generation
├── data_loader.py          # PyTorch Geometric dataset handling
├── model.py                # Hybrid GAT-HGT implementation
├── train.py                # Training loops & metrics
├── solver.py               # ML-enhanced backtracking solver
├── evaluation_analysis.py  # Performance analysis & visualization
├── config.py               # Configuration management
├── experiment_tracker_fixed.py # W&B experiment tracking
└── FullRun.py              # Main execution scripts

## 🧠 Technical Insights

### The Three Types of Moves
1. **Type 1**: Constraint-violating (immediately illegal)
2. **Type 2**: Constraint-legal but solution-invalid (the hard case!)
3. **Type 3**: Solution-valid moves (what we want to predict)

The core ML challenge is distinguishing Type 2 from Type 3 - moves that satisfy immediate constraints but lead to unsolvable states downstream.

### Model Features
- **Heterogeneous Edges**: Different attention mechanisms for different constraint types
- **Mid-sequence Global Context**: HGT transformer layer for global reasoning
- **Input Injection**: Raw features injected at multiple layers
- **Focal Loss**: Handles class imbalance in move predictions

## 🎛️ Configuration

Key parameters in `config.py`:
- Model: 6 GAT layers w/ 2 heads, 2 HGT layers w/ 4 heads, 128 hidden dim
- Training: Focal loss (α=0.25, γ=2.0), AdamW optimizer
- Data: 10% validation split, 512 batch size

## 📈 Monitoring

Experiment tracking with Weights & Biases:
- Training/validation metrics (loss, F1, top-1 accuracy)
- Gradient analysis for vanishing gradient detection
- Prediction samples and confidence distributions
- System resource monitoring

## 🔍 Analysis Tools

- **Board size effects**: Performance across different puzzle sizes
- **Game state analysis**: Accuracy vs. queens remaining
- **Spatial patterns**: Error heatmaps and position-based analysis
- **Solver comparison**: ML vs. traditional backtracking benchmarks

## 🎯 Learning Objectives

- Graph Neural Network design for constraint satisfaction
- Heterogeneous graph modeling
- ML-guided search algorithms
- Systematic evaluation and analysis
- Experiment tracking and reproducibility
