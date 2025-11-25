import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Set
import time
from dataclasses import dataclass
from collections import defaultdict

from model import HeteroGAT
from data_loader import build_heterogeneous_edge_index


@dataclass
class SolveResult:
    """Results from solving a queens puzzle."""
    success: bool
    solution: Optional[np.ndarray]
    queen_positions: Optional[List[Tuple[int, int]]]
    steps_taken: int
    backtracks: int
    solve_time: float
    decision_log: List[Dict]


class ModelEnabledQueensSolver:
    """Queens puzzle solver with smart backtracking and cycle detection."""

    def __init__(self, model: HeteroGAT, device: str = "cuda"):
        """Initialize solver with trained model."""
        self.model = model
        self.device = device
        self.max_regions = 11

        self.min_top_k = 3
        self.max_top_k = 6
        self.early_termination_threshold = -10.0

        self.model.eval()
        self._reset_tracking()

    def _reset_tracking(self):
        """Reset tracking variables for a new puzzle."""
        self.steps_taken = 0
        self.backtracks = 0
        self.decision_log = []
        self.visited_states = set()
        self.failed_moves = defaultdict(set)  # state_hash -> set of failed moves

    def solve_puzzle(self, region_board: np.ndarray, expected_solution: np.ndarray, verbose: bool = False) -> SolveResult:
        """Solve a queens puzzle using model guidance with smart backtracking."""
        start_time = time.time()
        n = region_board.shape[0]

        queen_board = np.zeros((n, n), dtype=int)
        placed_queens = []
        used_columns = set()
        used_regions = set()

        self._reset_tracking()
        edge_index_dict = self._build_edge_index(region_board)

        if verbose:
            print(f"Solving {n}x{n} puzzle with {len(np.unique(region_board))} regions")
            print("Enhanced solver: cycle detection + failed move memoization + dynamic top-k")

        success = self._solve_recursive(
            region_board, expected_solution, queen_board, placed_queens,
            used_columns, used_regions, edge_index_dict, verbose, current_depth=0
        )

        solve_time = time.time() - start_time

        if success:
            if verbose:
                print(f"\nSolver succeeded in {solve_time:.3f}s")
                print(f"Steps: {self.steps_taken}, Backtracks: {self.backtracks}")
                print(f"States explored: {len(self.visited_states)}")
                print(f"Failed moves cached: {sum(len(moves) for moves in self.failed_moves.values())}")

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
                print(f"\nSolver failed after {solve_time:.3f}s")
                print(f"Steps: {self.steps_taken}, Backtracks: {self.backtracks}")

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
                        used_regions: set, edge_index_dict: Dict, verbose: bool, current_depth: int = 0) -> bool:
        """Recursive solving with smart backtracking. Returns True if solution found."""
        n = region_board.shape[0]
        max_depth = n * 20

        if np.array_equal(queen_board, expected_solution):
            return True

        if current_depth > max_depth:
            if verbose:
                print(f"Depth limit {max_depth} reached, terminating")
            return False

        # Adaptive early termination based on progress
        progress_ratio = len(placed_queens) / n
        if current_depth > n * 5 and progress_ratio < 0.5:
            if verbose:
                print(f"Poor progress at depth {current_depth} (progress: {progress_ratio:.1%}), terminating")
            return False

        state_hash = self._hash_board_state(queen_board)
        if state_hash in self.visited_states:
            if verbose:
                print(f"Cycle detected at step {self.steps_taken}, skipping")
            return False

        self.visited_states.add(state_hash)
        self.steps_taken += 1

        legal_positions = self._get_legal_positions_enhanced(
            region_board, queen_board, edge_index_dict,
            used_columns, used_regions, placed_queens, verbose
        )

        # Filter out previously failed moves from this state
        filtered_positions = [
            (pos, logit) for pos, logit in legal_positions
            if pos not in self.failed_moves[state_hash]
        ]

        if self._is_state_unsolvable(filtered_positions, n, placed_queens):
            if verbose:
                print(f"UNSOLVABLE STATE detected at step {self.steps_taken}: No legal moves, {len(placed_queens)}/{n} queens placed")
                print(f"This will trigger multi-level backtracking to find alternative paths")
            self.visited_states.remove(state_hash)
            return False

        if not filtered_positions:
            if verbose:
                print(f"No untried legal positions at step {self.steps_taken}")
            self.visited_states.remove(state_hash)
            return False

        candidates_to_try = self._get_dynamic_top_k(filtered_positions)

        if candidates_to_try and candidates_to_try[0][1] < self.early_termination_threshold:
            if verbose:
                print(f"Early termination: best logit {candidates_to_try[0][1]:.3f} < {self.early_termination_threshold}")
            self.visited_states.remove(state_hash)
            return False

        for i, ((row, col), logit) in enumerate(candidates_to_try):
            if verbose:
                print(f"Step {self.steps_taken}: Trying position ({row}, {col}) with logit {logit:.3f} " +
                      f"(option {i+1}/{len(candidates_to_try)}, depth {current_depth})")

            queen_board[row, col] = 1
            placed_queens.append((row, col))
            used_columns.add(col)
            used_regions.add(region_board[row, col])

            self.decision_log.append({
                'step': self.steps_taken,
                'position': (row, col),
                'logit': float(logit),
                'num_legal_options': len(legal_positions),
                'num_filtered_options': len(filtered_positions),
                'candidate_rank': i + 1,
                'region': int(region_board[row, col]),
                'depth': current_depth
            })

            if self._solve_recursive(region_board, expected_solution, queen_board, placed_queens,
                                   used_columns, used_regions, edge_index_dict, verbose, current_depth + 1):
                return True

            if verbose:
                print(f"Backtracking from ({row}, {col}) at depth {current_depth}")

            self.backtracks += 1
            queen_board[row, col] = 0
            placed_queens.pop()
            used_columns.remove(col)
            used_regions.remove(region_board[row, col])

            self.failed_moves[state_hash].add((row, col))

        self.visited_states.remove(state_hash)
        return False

    def _hash_board_state(self, queen_board: np.ndarray) -> int:
        """Create unique hash for current queen placement."""
        return hash(queen_board.tobytes())

    def _get_dynamic_top_k(self, legal_positions: List[Tuple[Tuple[int, int], float]]) -> List[Tuple[Tuple[int, int], float]]:
        """Dynamically determine candidates to try based on confidence."""
        if not legal_positions:
            return []

        top_logit = legal_positions[0][1]

        # High confidence: try fewer candidates
        if top_logit > 0.5:
            num_candidates = min(3, len(legal_positions))
        else:
            num_candidates = min(self.max_top_k, len(legal_positions))

        num_candidates = max(self.min_top_k, num_candidates)
        num_candidates = min(num_candidates, len(legal_positions))

        return legal_positions[:num_candidates]

    def _validate_queen_placement(self, row: int, col: int, region_board: np.ndarray,
                                 queen_board: np.ndarray, used_columns: set, used_regions: set,
                                 placed_queens: List[Tuple[int, int]], verbose: bool = False) -> Tuple[bool, List[str]]:
        """Validate placement against all constraints. Returns (is_valid, violations)."""
        violations = []

        if col in used_columns:
            violations.append(f"column_{col}_occupied")

        region_id = region_board[row, col]
        if region_id in used_regions:
            violations.append(f"region_{region_id}_occupied")

        for q_row, q_col in placed_queens:
            if abs(row - q_row) == 1 and abs(col - q_col) == 1:
                violations.append(f"diagonal_conflict_with_({q_row},{q_col})")

        return len(violations) == 0, violations

    def _build_edge_index(self, region_board: np.ndarray) -> Dict[str, torch.Tensor]:
        """Build heterogeneous edge index for the puzzle (done once)."""
        edge_index_dict = build_heterogeneous_edge_index(region_board)

        for edge_type, edge_index in edge_index_dict.items():
            edge_index_dict[edge_type] = edge_index.to(self.device)

        return edge_index_dict

    def _get_legal_positions_enhanced(self, region_board: np.ndarray, queen_board: np.ndarray,
                                    edge_index_dict: Dict, used_columns: set, used_regions: set,
                                    placed_queens: List[Tuple[int, int]], verbose: bool = False) -> List[Tuple[Tuple[int, int], float]]:
        """Get legal positions with validation and model predictions, sorted by logit."""
        positions_by_logit, logits, _ = self._get_model_predictions(
            region_board, queen_board, edge_index_dict
        )

        n = region_board.shape[0]
        logits_np = logits.cpu().numpy().reshape(n, n)
        legal_positions = []

        for row, col in positions_by_logit:
            is_valid, violations = self._validate_queen_placement(
                row, col, region_board, queen_board, used_columns, used_regions, placed_queens, verbose
            )

            if is_valid:
                logit_value = logits_np[row, col]
                legal_positions.append(((row, col), logit_value))

        return legal_positions

    def _get_model_predictions(self, region_board: np.ndarray, queen_board: np.ndarray,
                             edge_index_dict: Dict) -> Tuple[List[Tuple[int, int]], torch.Tensor, float]:
        """Get model predictions for current board state. Returns (positions, logits, avg_confidence)."""
        n = region_board.shape[0]

        node_features = self._build_node_features(region_board, queen_board)
        node_features = node_features.to(self.device)

        with torch.no_grad():
            x_dict = {'cell': node_features}
            edge_index_dict_formatted = {
                ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
            }
            logits = self.model(x_dict, edge_index_dict_formatted)

        logits_np = logits.cpu().numpy().reshape(n, n)

        positions = [(r, c) for r in range(n) for c in range(n)]
        positions.sort(key=lambda pos: logits_np[pos[0], pos[1]], reverse=True)

        avg_confidence = torch.sigmoid(logits).mean().item()

        return positions, logits, avg_confidence

    def _build_node_features(self, region_board: np.ndarray, queen_board: np.ndarray) -> torch.Tensor:
        """Build node features: [row_coord, col_coord, region_onehot..., has_queen]."""
        n = region_board.shape[0]
        N2 = n * n

        coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)  # (N^2, 2)

        reg_onehot = np.zeros((N2, self.max_regions), dtype=np.float32)
        flat_ids = region_board.flatten()
        reg_onehot[np.arange(N2), flat_ids] = 1.0  # (N^2, 11)

        has_queen = queen_board.flatten()[:, None].astype(np.float32)  # (N^2, 1)

        features = np.hstack([coords, reg_onehot, has_queen])  # (N^2, 14)

        return torch.from_numpy(features)

    def _is_state_unsolvable(self, legal_positions: List[Tuple[Tuple[int, int], float]],
                            n: int, placed_queens: List[Tuple[int, int]]) -> bool:
        """Detect if current state cannot lead to a solution (triggers multi-level backtracking)."""
        queens_remaining = n - len(placed_queens)

        if len(legal_positions) == 0 and queens_remaining > 0:
            return True

        return False

    def _has_diagonal_conflict(self, row: int, col: int, placed_queens: List[Tuple[int, int]]) -> bool:
        """Check if position conflicts diagonally with any placed queen."""
        for q_row, q_col in placed_queens:
            if abs(row - q_row) == 1 and abs(col - q_col) == 1:
                return True
        return False


def load_solver(model_path: str, device: str = "cuda") -> ModelEnabledQueensSolver:
    """Load a trained model and create an enhanced solver."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['config_dict']

    is_hrm = model_config.get('model_type') == 'HRM' or 'n_cycles' in model_config

    if is_hrm:
        from model import HRM
        model = HRM(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            gat_heads=model_config.get('gat_heads', 2),
            hgt_heads=model_config.get('hgt_heads', 6),
            dropout=model_config.get('dropout', 0.2),
            use_batch_norm=True,
            n_cycles=model_config.get('n_cycles', 2),
            t_micro=model_config.get('t_micro', 2),
            use_input_injection=model_config.get('use_input_injection', True),
            z_init=model_config.get('z_init', 'zeros'),
        )
        print(f"Loaded HRM solver (cycles={model_config.get('n_cycles', 2)}, t_micro={model_config.get('t_micro', 2)})")
    else:
        from model import HeteroGAT
        model = HeteroGAT(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            layer_count=model_config['layer_count'],
            dropout=model_config['dropout'],
            gat_heads=model_config['gat_heads'],
            hgt_heads=model_config['hgt_heads']
        )
        print(f"Loaded HeteroGAT solver")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Model config: {model_config}")

    return ModelEnabledQueensSolver(model, device)


def test_solver_on_puzzle(solver: ModelEnabledQueensSolver, region_board: np.ndarray,
                         expected_solution: Optional[np.ndarray] = None, verbose: bool = True):
    """Test solver on a specific puzzle and print results."""
    print(f"Testing solver on {region_board.shape[0]}x{region_board.shape[0]} puzzle")
    print(f"Regions: {sorted(np.unique(region_board))}")

    result = solver.solve_puzzle(region_board, expected_solution, verbose=verbose)

    print(f"\n{'='*50}")
    print(f"SOLVE RESULT")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    print(f"Time: {result.solve_time:.3f}s")
    print(f"Steps: {result.steps_taken}")
    print(f"Backtracks: {result.backtracks}")
    print(f"Efficiency: {result.backtracks/result.steps_taken:.1%} backtrack rate" if result.steps_taken > 0 else "No steps taken")

    if result.success:
        print(f"Queen positions: {result.queen_positions}")
        print(f"\nSolution board:")
        print(result.solution)

        if verify_solution(region_board, result.solution):
            print("Solution verified!")
        else:
            print("Solution verification failed!")

        if expected_solution is not None:
            matches_expected = np.array_equal(result.solution, expected_solution)
            print(f"Matches expected solution: {matches_expected}")

    return result


def verify_solution(region_board: np.ndarray, label_board: np.ndarray) -> bool:
    """Verify that a solution satisfies all queens constraints."""
    n = region_board.shape[0]
    queen_positions = [(r, c) for r in range(n) for c in range(n) if label_board[r, c] == 1]

    if len(queen_positions) != n:
        return False

    used_cols = set()
    for _, col in queen_positions:
        if col in used_cols:
            return False
        used_cols.add(col)

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


@dataclass
class TraditionalSolveResult:
    """Results from solving a queens puzzle using traditional backtracking."""
    success: bool
    solution: Optional[np.ndarray]
    queen_positions: Optional[List[Tuple[int, int]]]
    steps_taken: int
    backtracks: int
    solve_time: float
    decision_log: List[Dict]

def solve_queens_with_metrics(region, verbose=False):
    """Solve queens puzzle with traditional backtracking and performance tracking."""
    start_time = time.time()
    region = np.asarray(region)
    n, m = region.shape
    assert n == m, "Board must be square"

    steps_taken = 0
    backtracks = 0
    decision_log = []
    constraint_checks = 0

    columns_used = set()
    regions_used = set()
    positions = []

    def backtrack(row):
        nonlocal steps_taken, backtracks, constraint_checks

        if row == n:
            return True

        for col in range(n):
            constraint_checks += 1
            reg_id = region[row, col]

            constraints_violated = []

            if col in columns_used:
                constraints_violated.append('column_conflict')

            if reg_id in regions_used:
                constraints_violated.append('region_conflict')

            diagonal_conflict = any(abs(r - row) == 1 and abs(c - col) == 1 for r, c in positions)
            if diagonal_conflict:
                constraints_violated.append('diagonal_conflict')

            if constraints_violated:
                if verbose:
                    print(f"Constraint check {constraint_checks}: Position ({row}, {col}) invalid - {', '.join(constraints_violated)}")
                continue

            steps_taken += 1
            if verbose:
                print(f"Step {steps_taken}: Placing queen at ({row}, {col}) in region {reg_id}")

            decision_log.append({
                'step': steps_taken,
                'position': (row, col),
                'region': int(reg_id),
                'attempt_type': 'placement',
                'constraints_violated': [],
                'is_valid_move': True
            })

            columns_used.add(col)
            regions_used.add(reg_id)
            positions.append((row, col))

            if backtrack(row + 1):
                return True

            backtracks += 1
            if verbose:
                print(f"Backtrack #{backtracks}: Removing queen from ({row}, {col})")

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

    success = backtrack(0)
    solve_time = time.time() - start_time

    if success:
        board = np.zeros((n, n), dtype=int)
        for r, c in positions:
            board[r, c] = 1

        if verbose:
            print(f"\nTraditional solver succeeded in {solve_time:.3f}s")
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
            print(f"\nTraditional solver failed after {solve_time:.3f}s")
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

def compare_solvers(region_board, expected_solution ,ml_solver=None, verbose=False):
    """Compare traditional and ML-enabled solvers on the same puzzle."""
    print(f"Comparing solvers on {region_board.shape[0]}x{region_board.shape[0]} puzzle")
    print(f"Regions: {sorted(np.unique(region_board))}")

    print("\nRunning traditional backtracking solver...")
    traditional_result = solve_queens_with_metrics(region_board, verbose=verbose)

    ml_result = None
    if ml_solver is not None:
        print("\nRunning ML-enabled solver...")
        try:
            ml_result = ml_solver.solve_puzzle(region_board, expected_solution, verbose=verbose)
        except Exception as e:
            print(f"ML solver error: {e}")
            ml_result = None

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

    print(f"\n{'='*60}")
    print("SOLVER COMPARISON RESULTS")
    print(f"{'='*60}")

    trad = comparison['traditional']
    print(f"Traditional: {'Success' if trad['success'] else 'Failed'} | "
          f"Time: {trad['solve_time']:.3f}s | "
          f"Steps: {trad['steps_taken']} | "
          f"Backtracks: {trad['backtracks']} ({trad['backtrack_rate']:.1%})")

    if 'ml' in comparison:
        ml = comparison['ml']
        print(f"ML-Enabled:  {'Success' if ml['success'] else 'Failed'} | "
              f"Time: {ml['solve_time']:.3f}s | "
              f"Steps: {ml['steps_taken']} | "
              f"Backtracks: {ml['backtracks']} ({ml['backtrack_rate']:.1%})")

        if 'improvement' in comparison:
            imp = comparison['improvement']
            print(f"\nML Improvements:")
            print(f"  Time: {imp['time_improvement_pct']:+.1f}%")
            print(f"  Steps: {imp['steps_improvement_pct']:+.1f}%")
            print(f"  Backtracks: {imp['backtrack_improvement_pct']:+.1f}%")
            print(f"  Overall: {'ML is better!' if imp['ml_is_better'] else 'Traditional is better'}")

    return comparison, traditional_result, ml_result

def analyze_solve_patterns(solve_result):
    """Analyze patterns in the solving process from decision log."""
    if not solve_result.decision_log:
        return {"error": "No decision log available"}

    log = solve_result.decision_log

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

def demo_enhanced_traditional_solver():
    """Demo function showing enhanced traditional solver usage."""
    print("Demo: Enhanced traditional solver with performance metrics")
    print("Usage:")
    print("result = solve_queens_with_metrics(your_region_array, verbose=True)")
    print("analysis = analyze_solve_patterns(result)")
    print("comparison = compare_solvers(region_board, ml_solver=your_ml_solver)")
