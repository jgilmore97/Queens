import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
import time
from dataclasses import dataclass

from model import HeteroGAT
from data_loader import build_heterogeneous_edge_index


@dataclass
class SolveResult:
    """Results from solving a queens puzzle."""
    success: bool
    solution: Optional[np.ndarray]  # Board with 1s where queens are placed
    queen_positions: Optional[List[Tuple[int, int]]]
    steps_taken: int
    backtracks: int
    solve_time: float
    decision_log: List[Dict]

class ModelEnabledQueensSolver:
    """
    Queens puzzle solver that uses a trained GNN model to guide placement decisions.
    """
    
    def __init__(self, model: HeteroGAT, device: str = "cuda"):
        """
        Initialize solver with trained model.
        
        Args:
            model: Trained HeteroGAT model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.max_regions = 11 
        self.top_k_candidates = 5 
        self.early_termination_threshold = -10.0  

        self.model.eval()
        
        # Tracking for current solve
        self.steps_taken = 0
        self.backtracks = 0
        self.decision_log = []
        
    def solve_puzzle(self, region_board: np.ndarray, expected_solution: np.ndarray, verbose: bool = False) -> SolveResult:
        """
        Solve a queens puzzle using model guidance with backtracking.
        
        Args:
            region_board: n×n numpy array with integer region IDs
            verbose: Whether to print step-by-step progress
            
        Returns:
            SolveResult with solution and metadata
        """
        start_time = time.time()
        n = region_board.shape[0]
        
        # Initialize solving state
        queen_board = np.zeros((n, n), dtype=int)  # 0 = empty, 1 = queen
        placed_queens = []
        used_columns = set()
        used_regions = set()
        
        # Reset tracking
        self.steps_taken = 0
        self.backtracks = 0
        self.decision_log = []
        
        # Build edge index once
        edge_index_dict = self._build_edge_index(region_board)
        
        if verbose:
            print(f"Solving {n}×{n} puzzle with {len(np.unique(region_board))} regions")
        
        # Start recursive solving
        success = self._solve_recursive(
            region_board, expected_solution, queen_board, placed_queens,
            used_columns, used_regions, edge_index_dict, verbose
        )
        
        solve_time = time.time() - start_time
        
        if success:
            if verbose:
                print(f"\n✅ Puzzle solved in {solve_time:.3f}s with {self.steps_taken} steps and {self.backtracks} backtracks")
            
            return SolveResult(
                success=True,
                solution=queen_board.copy(),
                queen_positions=placed_queens.copy(),
                steps_taken=self.steps_taken,
                backtracks=self.backtracks,
                solve_time=solve_time,
                decision_log=self.decision_log.copy()
            )
        else:
            if verbose:
                print(f"\n❌ Puzzle failed after {solve_time:.3f}s with {self.steps_taken} steps and {self.backtracks} backtracks")
            
            return SolveResult(
                success=False,
                solution=None,
                queen_positions=None,
                steps_taken=self.steps_taken,
                backtracks=self.backtracks,
                solve_time=solve_time,
                decision_log=self.decision_log.copy()
            )
    
    def _solve_recursive(self, region_board: np.ndarray, expected_solution: np.ndarray, queen_board: np.ndarray, 
                        placed_queens: List[Tuple[int, int]], used_columns: set, 
                        used_regions: set, edge_index_dict: Dict, verbose: bool) -> bool:
        """
        Recursive solving function with backtracking.
        
        Returns:
            True if solution found, False if this path failed
        """
        n = region_board.shape[0]
        
        # Base case: queens_board == expected_solution
        if np.array_equal(queen_board, expected_solution):
            return True
        
        self.steps_taken += 1
        
        # Get model predictions for current state
        legal_positions = self._get_legal_positions(
            region_board, queen_board, edge_index_dict, 
            used_columns, used_regions, placed_queens
        )
        
        # Early termination: no legal moves or all logits too low
        if not legal_positions:
            if verbose:
                print(f"No legal positions at step {self.steps_taken}")
            return False
        
        max_logit = max(logit for _, logit in legal_positions)
        if max_logit < self.early_termination_threshold:
            if verbose:
                print(f"Early termination: max logit {max_logit:.3f} < {self.early_termination_threshold}")
            return False
        
        # Try top-k candidates
        candidates_to_try = legal_positions[:self.top_k_candidates]
        
        for i, ((row, col), logit) in enumerate(candidates_to_try):
            if verbose:
                print(f"Step {self.steps_taken}: Trying position ({row}, {col}) with logit {logit:.3f} (option {i+1}/{len(candidates_to_try)})")
            
            # Place queen
            queen_board[row, col] = 1
            placed_queens.append((row, col))
            used_columns.add(col)
            used_regions.add(region_board[row, col])
            
            # Log this decision
            self.decision_log.append({
                'step': self.steps_taken,
                'position': (row, col),
                'logit': float(logit),
                'num_legal_options': len(legal_positions),
                'candidate_rank': i + 1,
                'region': int(region_board[row, col])
            })
            
            # Recurse
            if self._solve_recursive(region_board, expected_solution, queen_board, placed_queens, 
                                   used_columns, used_regions, edge_index_dict, verbose):
                return True  # Found solution in this path
            
            # Backtrack: remove queen and restore state
            if verbose:
                print(f"Backtracking from ({row}, {col})")
            
            self.backtracks += 1
            queen_board[row, col] = 0
            placed_queens.pop()
            used_columns.remove(col)
            used_regions.remove(region_board[row, col])
        
        # All candidates failed
        return False
    
    def _build_edge_index(self, region_board: np.ndarray) -> Dict[str, torch.Tensor]:
        """Build heterogeneous edge index for the puzzle (done once)."""
        edge_index_dict = build_heterogeneous_edge_index(region_board)
        
        for edge_type, edge_index in edge_index_dict.items():
            edge_index_dict[edge_type] = edge_index.to(self.device)
        
        return edge_index_dict
    
    def _get_legal_positions(self, region_board: np.ndarray, queen_board: np.ndarray, 
                           edge_index_dict: Dict, used_columns: set, used_regions: set, 
                           placed_queens: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get legal positions sorted by model confidence (highest logit first).
        
        Returns:
            List of ((row, col), logit) tuples sorted by logit (highest first)
        """
        # Get model predictions
        positions_by_logit, logits, _ = self._get_model_predictions(
            region_board, queen_board, edge_index_dict
        )
        
        # Filter to legal positions
        n = region_board.shape[0]
        logits_np = logits.cpu().numpy().reshape(n, n)
        legal_positions = []
        
        for row, col in positions_by_logit:
            # Check constraints
            if col in used_columns:
                continue
            if region_board[row, col] in used_regions:
                continue
            
            # Check diagonal adjacency to existing queens
            if self._has_diagonal_conflict(row, col, placed_queens):
                continue
            
            # Valid position - add with its logit
            logit_value = logits_np[row, col]
            legal_positions.append(((row, col), logit_value))
        
        return legal_positions
    
    def _get_model_predictions(self, region_board: np.ndarray, queen_board: np.ndarray, 
                             edge_index_dict: Dict) -> Tuple[List[Tuple[int, int]], torch.Tensor, float]:
        """
        Get model predictions for current board state.
        
        Returns:
            - List of (row, col) positions sorted by logit (highest first)
            - Raw logits tensor
            - Average model confidence (sigmoid of logits)
        """
        n = region_board.shape[0]
        
        # Build node features
        node_features = self._build_node_features(region_board, queen_board)
        node_features = node_features.to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            x_dict = {'cell': node_features}
            edge_index_dict_formatted = {
                ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
            }
            logits = self.model(x_dict, edge_index_dict_formatted)
        
        # Convert to numpy and reshape to board
        logits_np = logits.cpu().numpy().reshape(n, n)
        
        # Get all positions sorted by logit (highest first)
        positions = [(r, c) for r in range(n) for c in range(n)]
        positions.sort(key=lambda pos: logits_np[pos[0], pos[1]], reverse=True)
        
        # Calculate average confidence
        avg_confidence = torch.sigmoid(logits).mean().item()
        
        return positions, logits, avg_confidence
    
    def _build_node_features(self, region_board: np.ndarray, queen_board: np.ndarray) -> torch.Tensor:
        """
        Build node features for current board state.
        
        Features: [row_coord, col_coord, region_onehot..., has_queen]
        """
        n = region_board.shape[0]
        N2 = n * n
        
        # 1. Scaled coordinates
        coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)  # (N², 2)
        
        # 2. One-hot region encoding
        reg_onehot = np.zeros((N2, self.max_regions), dtype=np.float32)
        flat_ids = region_board.flatten()
        reg_onehot[np.arange(N2), flat_ids] = 1.0  # (N², 11)
        
        # 3. Has-queen flag
        has_queen = queen_board.flatten()[:, None].astype(np.float32)  # (N², 1)
        
        # Combine features
        features = np.hstack([coords, reg_onehot, has_queen])  # (N², 2 + 11 + 1 = 14)
        
        return torch.from_numpy(features)
    
    def _has_diagonal_conflict(self, row: int, col: int, placed_queens: List[Tuple[int, int]]) -> bool:
        """Check if position conflicts diagonally with any placed queen."""
        for q_row, q_col in placed_queens:
            # Check immediate diagonal adjacency (not full diagonal like chess)
            if abs(row - q_row) == 1 and abs(col - q_col) == 1:
                return True
        return False


def load_solver(model_path: str, device: str = "cuda") -> ModelEnabledQueensSolver:
    """
    Load a trained model and create a solver.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to run on
        
    Returns:
        Initialized solver
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['config_dict']
    
    # Create model
    model = HeteroGAT(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        layer_count=model_config['layer_count'],
        dropout=model_config['dropout'],
        heads=model_config['heads']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model config: {model_config}")
    
    return ModelEnabledQueensSolver(model, device)


def test_solver_on_puzzle(solver: ModelEnabledQueensSolver, region_board: np.ndarray,
                         expected_solution: Optional[np.ndarray] = None, verbose: bool = True):
    """
    Test solver on a specific puzzle and print results.
    
    Args:
        solver: Initialized solver
        region_board: Puzzle to solve
        expected_solution: Known solution for verification (optional)
        verbose: Whether to print detailed progress
    """
    print(f"Testing solver on {region_board.shape[0]}×{region_board.shape[0]} puzzle")
    print(f"Regions: {sorted(np.unique(region_board))}")
    
    # Solve puzzle
    result = solver.solve_puzzle(region_board, expected_solution, verbose=verbose)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"SOLVE RESULT")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    print(f"Time: {result.solve_time:.3f}s")
    print(f"Steps: {result.steps_taken}")
    print(f"Backtracks: {result.backtracks}")
    
    if result.success:
        print(f"Queen positions: {result.queen_positions}")
        print(f"\nSolution board:")
        print(result.solution)
        
        # Verify solution manually
        if verify_solution(region_board, result.solution):
            print("✅ Solution verified!")
        else:
            print("❌ Solution verification failed!")
        
        # Compare to expected if provided
        if expected_solution is not None:
            matches_expected = np.array_equal(result.solution, expected_solution)
            print(f"Matches expected solution: {matches_expected}")
    
    # Print decision log summary
    print(f"\nDecision log summary:")
    print(f"Total decisions logged: {len(result.decision_log)}")
    if result.decision_log:
        successful_decisions = [d for d in result.decision_log if d['candidate_rank'] == 1]
        print(f"First-choice successes: {len(successful_decisions)}/{len(result.decision_log)}")
    
    return result


def verify_solution(region_board: np.ndarray, solution_board: np.ndarray) -> bool:
    """
    Verify that a solution satisfies all queens constraints.
    
    Args:
        region_board: Original puzzle regions
        solution_board: Proposed solution with 1s where queens are placed
        
    Returns:
        True if solution is valid
    """
    n = region_board.shape[0]
    queen_positions = [(r, c) for r in range(n) for c in range(n) if solution_board[r, c] == 1]
    
    # Check we have exactly n queens
    if len(queen_positions) != n:
        return False
    
    # Check column constraint
    used_cols = set()
    for _, col in queen_positions:
        if col in used_cols:
            return False
        used_cols.add(col)
    
    # Check region constraint
    used_regions = set()
    for row, col in queen_positions:
        region = region_board[row, col]
        if region in used_regions:
            return False
        used_regions.add(region)
    
    # Check diagonal constraint (immediate adjacency only)
    for i, (r1, c1) in enumerate(queen_positions):
        for j, (r2, c2) in enumerate(queen_positions):
            if i != j and abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                return False
    
    return True


# ------------------------------------------------------------------------------
@dataclass
class TraditionalSolveResult:
    """Results from solving a queens puzzle using traditional backtracking."""
    success: bool
    solution: Optional[np.ndarray]  # Board with 1s where queens are placed
    queen_positions: Optional[List[Tuple[int, int]]]  # List of (row, col) positions
    steps_taken: int
    backtracks: int
    solve_time: float
    decision_log: List[Dict]  # Log of decisions at each step

def solve_queens_with_metrics(region, verbose=False):
    """
    Enhanced solve_queens with performance tracking.
    
    Args:
        region: NxN numpy array of region IDs (colors)
        verbose: Whether to print step-by-step progress
    
    Returns:
        TraditionalSolveResult with solution and performance metrics
    """
    start_time = time.time()
    region = np.asarray(region)
    n, m = region.shape
    assert n == m, "Board must be square"

    # Initialize tracking variables
    steps_taken = 0
    backtracks = 0
    decision_log = []
    
    columns_used = set()
    regions_used = set()
    positions = []

    def backtrack(row):
        nonlocal steps_taken, backtracks
        
        if row == n:
            return True
        
        for col in range(n):
            steps_taken += 1
            reg_id = region[row, col]
            
            # Log this decision attempt
            decision_log.append({
                'step': steps_taken,
                'position': (row, col),
                'region': int(reg_id),
                'attempt_type': 'placement_attempt',
                'constraints_violated': []
            })
            
            # Check constraints and track violations
            constraints_violated = []
            
            # 1) one queen per column
            if col in columns_used:
                constraints_violated.append('column_conflict')
            
            # 2) one queen per region
            if reg_id in regions_used:
                constraints_violated.append('region_conflict')
            
            # 3) no diagonal adjacency
            diagonal_conflict = any(abs(r - row) == 1 and abs(c - col) == 1 for r, c in positions)
            if diagonal_conflict:
                constraints_violated.append('diagonal_conflict')
            
            # Update decision log with constraint violations
            decision_log[-1]['constraints_violated'] = constraints_violated
            decision_log[-1]['is_valid_move'] = len(constraints_violated) == 0
            
            # Skip if any constraints violated
            if constraints_violated:
                if verbose:
                    print(f"Step {steps_taken}: Position ({row}, {col}) invalid - {', '.join(constraints_violated)}")
                continue
            
            if verbose:
                print(f"Step {steps_taken}: Placing queen at ({row}, {col}) in region {reg_id}")
            
            # place queen
            columns_used.add(col)
            regions_used.add(reg_id)
            positions.append((row, col))
            
            # recurse to next row
            if backtrack(row + 1):
                return True
            
            # backtrack: this placement didn't lead to solution
            backtracks += 1
            if verbose:
                print(f"Backtrack #{backtracks}: Removing queen from ({row}, {col})")
            
            # Log the backtrack
            decision_log.append({
                'step': steps_taken,
                'position': (row, col),
                'region': int(reg_id),
                'attempt_type': 'backtrack',
                'constraints_violated': [],
                'is_valid_move': False
            })
            
            positions.pop()
            columns_used.remove(col)
            regions_used.remove(reg_id)
        
        return False

    # Run the backtracking algorithm
    success = backtrack(0)
    solve_time = time.time() - start_time
    
    if success:
        # build the board array
        board = np.zeros((n, n), dtype=int)
        for r, c in positions:
            board[r, c] = 1
        
        if verbose:
            print(f"\n✅ Traditional solver succeeded in {solve_time:.3f}s")
            print(f"Steps taken: {steps_taken}")
            print(f"Backtracks: {backtracks}")
            print(f"Efficiency: {backtracks/steps_taken:.1%} backtrack rate")
        
        return TraditionalSolveResult(
            success=True,
            solution=board,
            queen_positions=positions.copy(),
            steps_taken=steps_taken,
            backtracks=backtracks,
            solve_time=solve_time,
            decision_log=decision_log.copy()
        )
    else:
        if verbose:
            print(f"\n❌ Traditional solver failed after {solve_time:.3f}s")
            print(f"Steps taken: {steps_taken}")
            print(f"Backtracks: {backtracks}")
        
        return TraditionalSolveResult(
            success=False,
            solution=None,
            queen_positions=None,
            steps_taken=steps_taken,
            backtracks=backtracks,
            solve_time=solve_time,
            decision_log=decision_log.copy()
        )

def compare_solvers(region_board, ml_solver=None, verbose=False):
    """
    Compare traditional and ML-enabled solvers on the same puzzle.
    
    Args:
        region_board: nxn numpy array with region IDs
        ml_solver: Optional ModelEnabledQueensSolver instance
        verbose: Whether to print detailed progress
    
    Returns:
        dict with comparison results
    """
    print(f"Comparing solvers on {region_board.shape[0]}x{region_board.shape[0]} puzzle")
    print(f"Regions: {sorted(np.unique(region_board))}")
    
    # Run traditional solver
    print("\n🔧 Running traditional backtracking solver...")
    traditional_result = solve_queens_with_metrics(region_board, verbose=verbose)
    
    # Run ML solver if provided
    ml_result = None
    if ml_solver is not None:
        print("\n🤖 Running ML-enabled solver...")
        try:
            expected_solution = np.zeros_like(region_board)
            ml_result = ml_solver.solve_puzzle(region_board, expected_solution, verbose=verbose)
        except Exception as e:
            print(f"ML solver error: {e}")
            ml_result = None
    
    # Compare results
    comparison = {
        'traditional': {
            'success': traditional_result.success,
            'solve_time': traditional_result.solve_time,
            'steps_taken': traditional_result.steps_taken,
            'backtracks': traditional_result.backtracks,
            'backtrack_rate': traditional_result.backtracks / traditional_result.steps_taken if traditional_result.steps_taken > 0 else 0,
            'efficiency_score': traditional_result.steps_taken / (region_board.shape[0] ** 2) if traditional_result.success else float('inf')
        }
    }
    
    if ml_result is not None:
        comparison['ml'] = {
            'success': ml_result.success,
            'solve_time': ml_result.solve_time,
            'steps_taken': ml_result.steps_taken,
            'backtracks': ml_result.backtracks,
            'backtrack_rate': ml_result.backtracks / ml_result.steps_taken if ml_result.steps_taken > 0 else 0,
            'efficiency_score': ml_result.steps_taken / (region_board.shape[0] ** 2) if ml_result.success else float('inf')
        }
        
        # Calculate improvement metrics
        if traditional_result.success and ml_result.success:
            time_improvement = (traditional_result.solve_time - ml_result.solve_time) / traditional_result.solve_time
            steps_improvement = (traditional_result.steps_taken - ml_result.steps_taken) / traditional_result.steps_taken
            backtrack_improvement = (traditional_result.backtracks - ml_result.backtracks) / max(traditional_result.backtracks, 1)
            
            comparison['improvement'] = {
                'time_improvement_pct': time_improvement * 100,
                'steps_improvement_pct': steps_improvement * 100,
                'backtrack_improvement_pct': backtrack_improvement * 100,
                'ml_is_better': ml_result.solve_time < traditional_result.solve_time and ml_result.steps_taken <= traditional_result.steps_taken
            }
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("SOLVER COMPARISON RESULTS")
    print(f"{'='*60}")
    
    trad = comparison['traditional']
    print(f"Traditional: {'✅' if trad['success'] else '❌'} | "
          f"Time: {trad['solve_time']:.3f}s | "
          f"Steps: {trad['steps_taken']} | "
          f"Backtracks: {trad['backtracks']} ({trad['backtrack_rate']:.1%})")
    
    if 'ml' in comparison:
        ml = comparison['ml']
        print(f"ML-Enabled:  {'✅' if ml['success'] else '❌'} | "
              f"Time: {ml['solve_time']:.3f}s | "
              f"Steps: {ml['steps_taken']} | "
              f"Backtracks: {ml['backtracks']} ({ml['backtrack_rate']:.1%})")
        
        if 'improvement' in comparison:
            imp = comparison['improvement']
            print(f"\nML Improvements:")
            print(f"  Time: {imp['time_improvement_pct']:+.1f}%")
            print(f"  Steps: {imp['steps_improvement_pct']:+.1f}%")
            print(f"  Backtracks: {imp['backtrack_improvement_pct']:+.1f}%")
            print(f"  Overall: {'🎯 ML is better!' if imp['ml_is_better'] else '🤔 Traditional is better'}")
    
    return comparison, traditional_result, ml_result

def analyze_solve_patterns(solve_result):
    """
    Analyze patterns in the solving process from decision log.
    
    Args:
        solve_result: TraditionalSolveResult or ModelEnabledQueensSolver result
    
    Returns:
        dict with analysis results
    """
    if not solve_result.decision_log:
        return {"error": "No decision log available"}
    
    log = solve_result.decision_log
    
    # Count constraint violations
    constraint_violations = {
        'column_conflict': 0,
        'region_conflict': 0,
        'diagonal_conflict': 0
    }
    
    valid_moves = 0
    placement_attempts = 0
    backtracks = 0
    
    for decision in log:
        if decision['attempt_type'] == 'placement_attempt':
            placement_attempts += 1
            if decision['is_valid_move']:
                valid_moves += 1
            else:
                for violation in decision.get('constraints_violated', []):
                    constraint_violations[violation] += 1
        elif decision['attempt_type'] == 'backtrack':
            backtracks += 1
    
    analysis = {
        'total_decisions': len(log),
        'placement_attempts': placement_attempts,
        'valid_moves': valid_moves,
        'invalid_moves': placement_attempts - valid_moves,
        'backtracks': backtracks,
        'success_rate': valid_moves / placement_attempts if placement_attempts > 0 else 0,
        'constraint_violations': constraint_violations,
        'most_common_violation': max(constraint_violations.items(), key=lambda x: x[1])[0] if any(constraint_violations.values()) else None
    }
    
    return analysis

# Example usage function
def demo_enhanced_traditional_solver():
    """Demo function showing enhanced traditional solver usage."""
    print("Demo: Enhanced traditional solver with performance metrics")
    print("Usage:")
    print("result = solve_queens_with_metrics(your_region_array, verbose=True)")
    print("analysis = analyze_solve_patterns(result)")
    print("comparison = compare_solvers(region_board, ml_solver=your_ml_solver)")