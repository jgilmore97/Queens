import torch
import numpy as np
import json
from collections import defaultdict
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def evaluate_full_puzzle_capability(
    solver,
    dataset_path: str,
    output_log_path: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a model's capability to solve entire Queens puzzles perfectly.
    
    This function analyzes the model's performance on state-0 puzzles (empty boards),
    checking if it can solve the entire puzzle without mistakes in a single forward sequence.
    
    Args:
        solver: ModelEnabledQueensSolver instance with loaded model
        dataset_path: Path to dataset JSON file
        output_log_path: Optional path to save detailed logs as JSON
        verbose: Whether to print detailed progress and results
        
    Returns:
        Dictionary with comprehensive statistics and results
    """
    start_time = time.time()
    
    # Load dataset
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {dataset_path}")
    
    # Get solver device
    device = solver.device
    solver.model.eval()  # Set model to evaluation mode
    
    # Extract state-0 puzzles (empty boards)
    state_0_puzzles = []
    for puzzle in test_data:
        # Check for states format
        if 'step' in puzzle and puzzle['step'] == 0:
            state_0_puzzles.append(puzzle)
        # Check for standard puzzle format (no step field)
        elif 'step' not in puzzle:
            # This is already a full board, not a state
            state_0_puzzles.append(puzzle)
    
    if not state_0_puzzles:
        raise ValueError("No state-0 puzzles found in the dataset")
    
    if verbose:
        logger.info(f"Evaluating {len(state_0_puzzles)} puzzles for full-puzzle capability")
    
    # Statistics tracking
    stats = {
        "total_puzzles": len(state_0_puzzles),
        "perfect_solves": 0,
        "failed_solves": 0,
        "perfect_solve_rate": 0.0,
        "error_by_step": defaultdict(int),  # Step where first error occurred
        "error_by_board_size": defaultdict(lambda: {"total": 0, "errors": 0, "rate": 0.0}),
        "error_by_source": defaultdict(lambda: {"total": 0, "errors": 0}),
        "processing_time": 0.0,
        "puzzle_details": []
    }
    
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
            logger.error(f"Puzzle {puzzle_idx} has no solution data (label_board or solution_board) - skipping")
            stats["failed_extractions"] += 1
            continue
        
        source = puzzle.get('source', f"puzzle_{puzzle_idx}")
        
        # Update size statistics
        stats["error_by_board_size"][n]["total"] += 1
        stats["error_by_source"][source]["total"] += 1
        
        # Extract correct queen positions
        correct_positions = [(r, c) for r in range(n) for c in range(n) if expected_solution[r, c] == 1]
        
        # Initialize empty board for simulation
        queen_board = np.zeros((n, n), dtype=int)
        
        # Prepare edge indices once (optimization)
        try:
            edge_index_dict = solver._build_edge_index(region_board)
        except Exception as e:
            logger.error(f"Error building edge index for puzzle {puzzle_idx} ({source}): {e}")
            stats["failed_solves"] += 1
            stats["error_by_source"][source]["errors"] += 1
            stats["error_by_board_size"][n]["errors"] += 1
            continue
        
        # Simulation variables
        is_perfect = True
        first_error_step = None
        placement_log = []
        
        # Simulate the solving process step by step
        for step in range(n):
            # Build features
            try:
                node_features = solver._build_node_features(region_board, queen_board)
                node_features = node_features.to(device)
            except Exception as e:
                logger.error(f"Error building node features at step {step} for puzzle {puzzle_idx} ({source}): {e}")
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
                logger.error(f"Model inference error at step {step} for puzzle {puzzle_idx} ({source}): {e}")
                is_perfect = False
                first_error_step = step
                break
                
            # Get top prediction
            logits_np = logits.cpu().numpy().reshape(n, n)
            flat_logits = logits_np.flatten()
            top_idx = np.argmax(flat_logits)
            top_row, top_col = top_idx // n, top_idx % n
            top_logit = flat_logits[top_idx]
            
            # Get remaining correct positions
            remaining_correct = [pos for pos in correct_positions if queen_board[pos[0], pos[1]] == 0]
            
            # Check if top prediction is correct
            is_correct = (top_row, top_col) in remaining_correct
            
            # Log this placement
            placement_log.append({
                'step': step,
                'position': (int(top_row), int(top_col)),
                'logit': float(top_logit),
                'is_correct': bool(is_correct),
                'num_remaining_correct': len(remaining_correct)
            })
            
            # Check for errors
            if not is_correct and is_perfect:
                is_perfect = False
                first_error_step = step
                stats["error_by_step"][step] += 1
                stats["error_by_source"][source]["errors"] += 1
                stats["error_by_board_size"][n]["errors"] += 1
            
            # Place queen
            queen_board[top_row, top_col] = 1
            
            # Early stopping on first error if not verbose
            if not is_perfect and not verbose:
                break
        
        # Record results
        if is_perfect:
            stats["perfect_solves"] += 1
        else:
            stats["failed_solves"] += 1
        
        # Store detailed puzzle data
        puzzle_detail = {
            'puzzle_index': puzzle_idx,
            'source': source,
            'board_size': n,
            'is_perfect': is_perfect,
            'first_error_step': first_error_step,
            'placement_log': placement_log if verbose else None
        }
        
        stats["puzzle_details"].append(puzzle_detail)
        
        # Progress reporting
        if verbose and (puzzle_idx + 1) % 10 == 0:
            logger.info(f"Processed {puzzle_idx + 1}/{len(state_0_puzzles)} puzzles")
    
    # Calculate summary statistics
    stats["perfect_solve_rate"] = stats["perfect_solves"] / stats["total_puzzles"] if stats["total_puzzles"] > 0 else 0
    stats["processing_time"] = time.time() - start_time
    
    # Calculate error rates by board size
    for size, data in stats["error_by_board_size"].items():
        if data["total"] > 0:
            data["rate"] = data["errors"] / data["total"]
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("QUEENS PUZZLE EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total puzzles evaluated: {stats['total_puzzles']}")
        print(f"Perfect solves: {stats['perfect_solves']} ({stats['perfect_solve_rate']:.1%})")
        print(f"Failed solves: {stats['failed_solves']} ({1 - stats['perfect_solve_rate']:.1%})")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        # Error distribution by step
        if stats["error_by_step"]:
            print("\nError distribution by step:")
            for step in sorted(stats["error_by_step"].keys()):
                count = stats["error_by_step"][step]
                pct = count / stats["failed_solves"] if stats["failed_solves"] > 0 else 0
                print(f"  Step {step}: {count} errors ({pct:.1%} of all errors)")
        
        # Error rates by board size
        print("\nPerformance by board size:")
        for size in sorted(stats["error_by_board_size"].keys()):
            data = stats["error_by_board_size"][size]
            print(f"  {size}x{size}: {data['total'] - data['errors']}/{data['total']} perfect ({1 - data['rate']:.1%})")
        
        # Most problematic sources
        if len(stats["error_by_source"]) > 0:
            print("\nTop problematic sources:")
            problem_sources = sorted(
                [(s, d) for s, d in stats["error_by_source"].items() if d["errors"] > 0],
                key=lambda x: x[1]["errors"], 
                reverse=True
            )[:5]
            
            for source, data in problem_sources:
                print(f"  {source}: {data['errors']}/{data['total']} failures")
    
    # Save detailed log if requested
    if output_log_path:
        try:
            with open(output_log_path, 'w') as f:
                json.dump(stats, f, indent=2)
            if verbose:
                logger.info(f"Detailed evaluation log saved to: {output_log_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation log: {e}")
    
    return stats

def evaluate_puzzle(solver, region_board, expected_solution=None):
    """
    Evaluate a single puzzle with detailed step-by-step analysis.
    
    Args:
        solver: ModelEnabledQueensSolver instance
        region_board: n×n numpy array with region IDs
        expected_solution: Optional n×n numpy array with expected queen placements
                          If None, run without validation (inference only mode)
                          
    Returns:
        Dict with detailed solve data
    """
    n = region_board.shape[0]
    queen_board = np.zeros((n, n), dtype=int)
    
    # Check if we're validating against an expected solution
    if expected_solution is None:
        validate_solution = False
        logger.info("No expected solution provided - running in inference-only mode without validation")
    else:
        validate_solution = True
        if expected_solution.shape != region_board.shape:
            raise ValueError(f"Expected solution shape {expected_solution.shape} doesn't match region board {region_board.shape}")
    
    # Set model to evaluation mode
    solver.model.eval()
    device = solver.device
    
    # Build edge index once
    edge_index_dict = solver._build_edge_index(region_board)
    
    # Get correct positions if validating
    if validate_solution:
        correct_positions = [(r, c) for r in range(n) for c in range(n) if expected_solution[r, c] == 1]
        if len(correct_positions) != n:
            logger.warning(f"Expected solution has {len(correct_positions)} queens, not {n}")
    
    # Initialize result data
    result = {
        "board_size": n,
        "is_perfect": True if not validate_solution else None,
        "placements": [],
        "solve_time": 0
    }
    
    start_time = time.time()
    
    # Step through the solve process
    for step in range(n):
        # Build features
        node_features = solver._build_node_features(region_board, queen_board)
        node_features = node_features.to(device)
        
        # Get predictions
        with torch.no_grad():
            x_dict = {'cell': node_features}
            edge_index_dict_formatted = {
                ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
            }
            logits = solver.model(x_dict, edge_index_dict_formatted)
        
        # Process predictions
        logits_np = logits.cpu().numpy().reshape(n, n)
        flat_logits = logits_np.flatten()
        
        # Sort predictions
        sorted_indices = np.argsort(flat_logits)[::-1]
        sorted_positions = [(idx // n, idx % n) for idx in sorted_indices]
        sorted_logits = [float(flat_logits[idx]) for idx in sorted_indices]
        
        # If validating, check top prediction correctness
        if validate_solution:
            remaining_correct = [pos for pos in correct_positions if queen_board[pos[0], pos[1]] == 0]
            top_pos = sorted_positions[0]
            is_correct = top_pos in remaining_correct
            
            # Find rank of first correct prediction
            first_correct_rank = None
            first_correct_pos = None
            for i, pos in enumerate(sorted_positions):
                if pos in remaining_correct:
                    first_correct_rank = i + 1
                    first_correct_pos = pos
                    break
                    
            # Update perfect flag
            if not is_correct and result["is_perfect"]:
                result["is_perfect"] = False
                result["first_error_step"] = step
        else:
            is_correct = None
            first_correct_rank = None
            first_correct_pos = None
        
        # Get the top prediction
        top_row, top_col = sorted_positions[0]
        
        # Log step data
        step_data = {
            "step": step,
            "position": (int(top_row), int(top_col)),
            "logit": float(sorted_logits[0]),
            "is_correct": is_correct,
            "top_10_predictions": [
                {
                    "rank": i + 1,
                    "position": (int(r), int(c)),
                    "logit": float(l)
                } for i, ((r, c), l) in enumerate(zip(sorted_positions[:10], sorted_logits[:10]))
            ],
            "first_correct_rank": first_correct_rank,
            "first_correct_position": first_correct_pos
        }
        
        result["placements"].append(step_data)
        
        # Place the queen
        queen_board[top_row, top_col] = 1
    
    result["solve_time"] = time.time() - start_time
    
    # Final validation
    if validate_solution:
        result["matches_expected"] = np.array_equal(queen_board, expected_solution)
    
    # Generate final queen positions
    result["final_queen_positions"] = [
        (int(r), int(c)) for r in range(n) for c in range(n) if queen_board[r, c] == 1
    ]
    
    return result

def analyze_dataset_errors(stats):
    """
    Analyze the error patterns in evaluation statistics.
    
    Args:
        stats: Dict returned by evaluate_full_puzzle_capability
        
    Returns:
        Dict with error analysis
    """
    analysis = {
        "early_game_errors": 0,  # Steps 0-2
        "mid_game_errors": 0,    # Steps 3-6
        "late_game_errors": 0,   # Steps 7+
        "problematic_sizes": [],
        "problematic_sources": [],
        "error_analysis": "No errors to analyze" if stats["failed_solves"] == 0 else ""
    }
    
    # Count errors by game phase
    for step, count in stats["error_by_step"].items():
        if 0 <= step <= 2:
            analysis["early_game_errors"] += count
        elif 3 <= step <= 6:
            analysis["mid_game_errors"] += count
        else:
            analysis["late_game_errors"] += count
    
    # Find problematic board sizes
    for size, data in stats["error_by_board_size"].items():
        if data["rate"] > 0.1 and data["total"] >= 5:  # At least 10% error rate and enough samples
            analysis["problematic_sizes"].append({
                "size": size,
                "error_rate": data["rate"],
                "sample_count": data["total"]
            })
    
    # Find problematic sources
    for source, data in stats["error_by_source"].items():
        if data["errors"] > 0 and data["total"] > 0:
            error_rate = data["errors"] / data["total"]
            if error_rate > 0.5 and data["total"] >= 3:  # High error rate and multiple instances
                analysis["problematic_sources"].append({
                    "source": source,
                    "error_rate": error_rate,
                    "sample_count": data["total"]
                })
    
    # Sort by error rate
    analysis["problematic_sizes"].sort(key=lambda x: x["error_rate"], reverse=True)
    analysis["problematic_sources"].sort(key=lambda x: x["error_rate"], reverse=True)
    
    # Generate summary analysis
    if stats["failed_solves"] > 0:
        total_errors = sum(stats["error_by_step"].values())
        early_pct = analysis["early_game_errors"] / total_errors if total_errors > 0 else 0
        mid_pct = analysis["mid_game_errors"] / total_errors if total_errors > 0 else 0
        late_pct = analysis["late_game_errors"] / total_errors if total_errors > 0 else 0
        
        summary = []
        summary.append(f"Error distribution: Early {early_pct:.1%}, Mid {mid_pct:.1%}, Late {late_pct:.1%}")
        
        if early_pct > 0.6:
            summary.append("Most errors occur early in the game, suggesting fundamental puzzle ambiguity.")
        elif late_pct > 0.5:
            summary.append("Errors cluster in late stages, suggesting cascading/accumulated errors.")
        
        if analysis["problematic_sizes"]:
            sizes = [f"{s['size']}×{s['size']}" for s in analysis["problematic_sizes"][:3]]
            summary.append(f"Most problematic board sizes: {', '.join(sizes)}")
        
        if analysis["problematic_sources"]:
            sources = [s['source'] for s in analysis["problematic_sources"][:3]]
            summary.append(f"Most problematic puzzles: {', '.join(sources)}")
        
        analysis["error_analysis"] = " ".join(summary)
    
    return analysis

def print_detailed_error_report(stats, max_examples=5):
    """
    Print a detailed error report based on evaluation statistics.
    
    Args:
        stats: Dict returned by evaluate_full_puzzle_capability
        max_examples: Maximum number of examples to show for each category
    """
    if stats["failed_solves"] == 0:
        print("No errors found in the evaluation!")
        return
    
    print("\n" + "=" * 60)
    print("DETAILED ERROR REPORT")
    print("=" * 60)
    
    # First failure examples
    print("\nFirst Step Failures:")
    first_step_failures = [p for p in stats["puzzle_details"] 
                          if not p["is_perfect"] and p["first_error_step"] == 0]
    for i, failure in enumerate(first_step_failures[:max_examples]):
        print(f"  {failure['source']} ({failure['board_size']}×{failure['board_size']})")
    if len(first_step_failures) > max_examples:
        print(f"  ... and {len(first_step_failures) - max_examples} more")
    
    # Error distribution by step
    print("\nError distribution by step:")
    for step in sorted(stats["error_by_step"].keys()):
        count = stats["error_by_step"][step]
        pct = count / stats["failed_solves"] if stats["failed_solves"] > 0 else 0
        print(f"  Step {step}: {count} errors ({pct:.1%} of all errors)")
    
    # Print sample failures at each step
    for step in sorted(stats["error_by_step"].keys()):
        print(f"\nFailures at step {step} (sample):")
        step_failures = [p for p in stats["puzzle_details"] 
                         if not p["is_perfect"] and p["first_error_step"] == step]
        for i, failure in enumerate(step_failures[:max_examples]):
            print(f"  {failure['source']} ({failure['board_size']}×{failure['board_size']})")
    
    # Analysis by board size
    print("\nPerformance by board size:")
    for size in sorted(stats["error_by_board_size"].keys()):
        data = stats["error_by_board_size"][size]
        print(f"  {size}×{size}: {data['total'] - data['errors']}/{data['total']} perfect ({1 - data['rate']:.1%})")
    
    # Overall analysis
    analysis = analyze_dataset_errors(stats)
    print("\nError Analysis:")
    print(f"  {analysis['error_analysis']}")