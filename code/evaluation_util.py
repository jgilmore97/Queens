import torch
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional

def evaluate_full_puzzle_capability(
    solver,
    dataset_path: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model's capability to solve Queens puzzles perfectly.
    
    Simplified version focused on success/fail tracking and first error location.
    
    Args:
        solver: ModelEnabledQueensSolver instance with loaded model
        dataset_path: Path to dataset JSON file
        verbose: Whether to print results during evaluation
        
    Returns:
        Dictionary with evaluation statistics
    """
    start_time = time.time()
    
    # Load dataset
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    
    try:
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}
    
    # Get solver device
    device = solver.device
    solver.model.eval()
    
    # Extract state-0 puzzles (empty boards)
    state_0_puzzles = []
    for puzzle in test_data:
        if 'step' in puzzle and puzzle['step'] == 0:
            state_0_puzzles.append(puzzle)
        elif 'step' not in puzzle:
            # This is already a full board
            state_0_puzzles.append(puzzle)
    
    if not state_0_puzzles:
        print("No state-0 puzzles found in the dataset")
        return {}
    
    # Statistics tracking
    stats = {
        "total_puzzles": len(state_0_puzzles),
        "successful_solves": 0,
        "failed_solves": 0,
        "error_by_step": defaultdict(int),
        "error_by_board_size": defaultdict(lambda: {"total": 0, "errors": 0}),
        "processing_time": 0,
        "failed_puzzles": []
    }
    
    if verbose:
        print(f"Evaluating {len(state_0_puzzles)} puzzles...")
    
    # Process each puzzle
    for puzzle_idx, puzzle in enumerate(state_0_puzzles):
        # Extract puzzle information
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        
        # Handle different formats for expected solution
        if 'label_board' in puzzle:
            expected_solution = np.array(puzzle['label_board'])
        elif 'solution_board' in puzzle:
            expected_solution = np.array(puzzle['solution_board'])
        else:
            if verbose:
                print(f"Puzzle {puzzle_idx} has no solution data - skipping")
            continue
        
        source = puzzle.get('source', f"puzzle_{puzzle_idx}")
        
        # Update board size statistics
        stats["error_by_board_size"][n]["total"] += 1
        
        # Extract correct queen positions
        correct_positions = [(r, c) for r in range(n) for c in range(n) if expected_solution[r, c] == 1]
        
        # Initialize empty board for simulation
        queen_board = np.zeros((n, n), dtype=int)
        
        # Prepare edge indices
        try:
            edge_index_dict = solver._build_edge_index(region_board)
        except Exception as e:
            if verbose:
                print(f"Error building edge index for puzzle {puzzle_idx}: {e}")
            stats["failed_solves"] += 1
            stats["error_by_board_size"][n]["errors"] += 1
            continue
        
        # Simulation variables
        is_perfect = True
        first_error_step = None
        
        # Simulate the solving process
        for step in range(n):
            # Build features
            try:
                node_features = solver._build_node_features(region_board, queen_board)
                node_features = node_features.to(device)
            except Exception as e:
                if verbose:
                    print(f"Error building features at step {step} for puzzle {puzzle_idx}: {e}")
                is_perfect = False
                first_error_step = step
                break
            
            # Get model predictions
            try:
                with torch.no_grad():
                    x_dict = {'cell': node_features}
                    edge_index_dict_formatted = {
                        ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                        ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                        ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
                    }
                    logits = solver.model(x_dict, edge_index_dict_formatted)
            except Exception as e:
                if verbose:
                    print(f"Model inference error at step {step} for puzzle {puzzle_idx}: {e}")
                is_perfect = False
                first_error_step = step
                break
            
            # Get top prediction
            logits_np = logits.cpu().numpy().reshape(n, n)
            flat_logits = logits_np.flatten()
            top_idx = np.argmax(flat_logits)
            top_row, top_col = top_idx // n, top_idx % n
            
            # Get remaining correct positions
            remaining_correct = [pos for pos in correct_positions if queen_board[pos[0], pos[1]] == 0]
            
            # Check if top prediction is correct
            is_correct = (top_row, top_col) in remaining_correct
            
            # Record first error
            if not is_correct and is_perfect:
                is_perfect = False
                first_error_step = step
                stats["error_by_step"][step] += 1
                stats["error_by_board_size"][n]["errors"] += 1
                
                # Store basic info about this failed puzzle
                stats["failed_puzzles"].append({
                    "source": source,
                    "board_size": n,
                    "first_error_step": step
                })
                
                # Early stopping
                break
            
            # Place queen
            queen_board[top_row, top_col] = 1
        
        # Record results
        if is_perfect:
            stats["successful_solves"] += 1
        else:
            stats["failed_solves"] += 1
        
        if verbose and (puzzle_idx + 1) % 10 == 0:
            print(f"Processed {puzzle_idx + 1}/{len(state_0_puzzles)} puzzles")
    
    # Calculate success rate
    stats["success_rate"] = stats["successful_solves"] / stats["total_puzzles"] if stats["total_puzzles"] > 0 else 0
    stats["processing_time"] = time.time() - start_time
    
    # Print summary
    if verbose:
        print("\n" + "=" * 50)
        print("QUEENS PUZZLE EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total puzzles: {stats['total_puzzles']}")
        print(f"Successful solves: {stats['successful_solves']} ({stats['success_rate']:.1%})")
        print(f"Failed solves: {stats['failed_solves']} ({1-stats['success_rate']:.1%})")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        # Error distribution by step
        if stats["error_by_step"]:
            print("\nFirst error distribution by step:")
            for step in sorted(stats["error_by_step"].keys()):
                count = stats["error_by_step"][step]
                pct = count / stats["failed_solves"] if stats["failed_solves"] > 0 else 0
                print(f"  Step {step}: {count} errors ({pct:.1%} of all errors)")
        
        # Performance by board size
        print("\nPerformance by board size:")
        for size in sorted(stats["error_by_board_size"].keys()):
            data = stats["error_by_board_size"][size]
            success = data["total"] - data["errors"]
            total = data["total"]
            success_rate = success / total if total > 0 else 0
            print(f"  {size}×{size}: {success}/{total} successful ({success_rate:.1%})")
    
    return stats