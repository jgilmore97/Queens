import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    solved: bool
    solution: Optional[np.ndarray]
    time_ms: float
    guesses: int = 0
    failed_guesses: int = 0


def solve_backtracking(region: np.ndarray) -> SolverResult:
    n = region.shape[0]
    columns_used = set()
    regions_used = set()
    positions = []
    guesses = 0
    failed_guesses = 0

    start = time.perf_counter()

    def backtrack(row):
        nonlocal guesses, failed_guesses
        if row == n:
            return True
        for col in range(n):
            reg_id = region[row, col]
            if col in columns_used or reg_id in regions_used:
                continue
            if any(abs(r - row) == 1 and abs(c - col) == 1 for r, c in positions):
                continue

            guesses += 1
            columns_used.add(col)
            regions_used.add(reg_id)
            positions.append((row, col))

            if backtrack(row + 1):
                return True

            failed_guesses += 1
            positions.pop()
            columns_used.remove(col)
            regions_used.remove(reg_id)
        return False

    solved = backtrack(0)
    elapsed = (time.perf_counter() - start) * 1000

    if solved:
        board = np.zeros((n, n), dtype=int)
        for r, c in positions:
            board[r, c] = 1
        return SolverResult(True, board, elapsed, guesses, failed_guesses)
    return SolverResult(False, None, elapsed, guesses, failed_guesses)


def solve_ac3_backtracking(region: np.ndarray) -> SolverResult:
    n = region.shape[0]
    start = time.perf_counter()
    guesses = 0
    failed_guesses = 0

    def get_initial_domains():
        return [set(range(n)) for _ in range(n)]

    def propagate(domains, placed):
        changed = True
        while changed:
            changed = False
            used_cols = {col for _, col in placed}
            used_regions = {region[r, c] for r, c in placed}
            placed_positions = set(placed)

            for row in range(n):
                if any(r == row for r, _ in placed):
                    continue

                to_remove = set()
                for col in domains[row]:
                    if col in used_cols:
                        to_remove.add(col)
                        continue
                    if region[row, col] in used_regions:
                        to_remove.add(col)
                        continue
                    for pr, pc in placed_positions:
                        if abs(pr - row) == 1 and abs(pc - col) == 1:
                            to_remove.add(col)
                            break

                if to_remove:
                    domains[row] -= to_remove
                    changed = True

                if len(domains[row]) == 0:
                    return False

        return True

    def backtrack(row, domains, placed):
        nonlocal guesses, failed_guesses

        if row == n:
            return placed.copy()

        unplaced_rows = [r for r in range(n) if not any(pr == r for pr, _ in placed)]
        if not unplaced_rows:
            return placed.copy()

        next_row = min(unplaced_rows, key=lambda r: len(domains[r]))

        if len(domains[next_row]) == 0:
            return None

        for col in sorted(domains[next_row]):
            guesses += 1
            new_placed = placed + [(next_row, col)]
            new_domains = [d.copy() for d in domains]
            new_domains[next_row] = {col}

            if propagate(new_domains, new_placed):
                result = backtrack(row + 1, new_domains, new_placed)
                if result is not None:
                    return result

            failed_guesses += 1

        return None

    domains = get_initial_domains()
    propagate(domains, [])

    result = backtrack(0, domains, [])
    elapsed = (time.perf_counter() - start) * 1000

    if result:
        board = np.zeros((n, n), dtype=int)
        for r, c in result:
            board[r, c] = 1
        return SolverResult(True, board, elapsed, guesses, failed_guesses)

    return SolverResult(False, None, elapsed, guesses, failed_guesses)


def solve_ortools(region: np.ndarray) -> SolverResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        logger.warning("OR-Tools not installed. Run: pip install ortools")
        return SolverResult(False, None, 0.0, 0, 0)

    n = region.shape[0]
    start = time.perf_counter()

    model = cp_model.CpModel()
    queens = [[model.NewBoolVar(f'q_{i}_{j}') for j in range(n)] for i in range(n)]

    for i in range(n):
        model.AddExactlyOne(queens[i])
    for j in range(n):
        model.AddExactlyOne([queens[i][j] for i in range(n)])

    region_cells = {}
    for i in range(n):
        for j in range(n):
            reg = region[i, j]
            if reg not in region_cells:
                region_cells[reg] = []
            region_cells[reg].append(queens[i][j])
    for reg_id, cells in region_cells.items():
        model.AddExactlyOne(cells)

    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    model.AddAtMostOne([queens[i][j], queens[ni][nj]])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0

    status = solver.Solve(model)
    elapsed = (time.perf_counter() - start) * 1000

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        board = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if solver.Value(queens[i][j]):
                    board[i, j] = 1
        return SolverResult(
            True, board, elapsed,
            guesses=int(solver.NumBranches()),
            failed_guesses=int(solver.NumConflicts())
        )

    return SolverResult(
        False, None, elapsed,
        guesses=int(solver.NumBranches()),
        failed_guesses=int(solver.NumConflicts())
    )


def solve_neural(region: np.ndarray, solver) -> SolverResult:
    start = time.perf_counter()
    puzzle = {'region': region.tolist()}
    try:
        solution, _ = solver.solve_puzzle(puzzle, batch_placement=True)
        elapsed = (time.perf_counter() - start) * 1000
        solved = verify_solution(region, solution)
        return SolverResult(solved, solution, elapsed, guesses=0)
    except Exception as e:
        logger.error(f"Neural solver failed: {e}")
        elapsed = (time.perf_counter() - start) * 1000
        return SolverResult(False, None, elapsed, guesses=0)


def verify_solution(region: np.ndarray, solution: np.ndarray) -> bool:
    n = region.shape[0]
    queens = list(zip(*np.where(solution == 1)))

    if len(queens) != n:
        return False

    rows = [r for r, c in queens]
    cols = [c for r, c in queens]
    regions = [region[r, c] for r, c in queens]

    if len(set(rows)) != n or len(set(cols)) != n or len(set(regions)) != n:
        return False

    for i, (r1, c1) in enumerate(queens):
        for r2, c2 in queens[i+1:]:
            if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                return False

    return True


@dataclass
class ComparisonResult:
    solver_name: str
    puzzles_solved: int
    total_puzzles: int
    solve_rate: float
    avg_time_ms: float
    median_time_ms: float
    total_guesses: int
    avg_guesses_per_puzzle: float
    total_failed_guesses: int
    avg_failed_per_puzzle: float


def compare_solvers(
    puzzles: List[Dict],
    model_path: Optional[str] = None,
    device: str = "cpu",
    include_ortools: bool = True,
) -> Dict[str, ComparisonResult]:
    solvers = {
        'backtracking': solve_backtracking,
        'ac3': solve_ac3_backtracking,
    }

    if include_ortools:
        try:
            from ortools.sat.python import cp_model  # noqa: F401
            solvers['ortools'] = solve_ortools
        except ImportError:
            logger.info("OR-Tools not available, skipping")

    if model_path:
        try:
            from queens_solver.evaluation.solver import Solver
            neural_solver = Solver(model_path, device=device)
            solvers['neural'] = lambda r: solve_neural(r, neural_solver)
        except Exception as e:
            logger.warning(f"Could not load neural model: {e}")

    results = {}

    for solver_name, solver_fn in solvers.items():
        logger.info(f"Running {solver_name}...")
        solve_results = []

        for puzzle in puzzles:
            region = np.array(puzzle['region'])
            solve_results.append(solver_fn(region))

        solved = sum(1 for r in solve_results if r.solved)
        times = [r.time_ms for r in solve_results]
        total_guesses = sum(r.guesses for r in solve_results)
        total_failed = sum(r.failed_guesses for r in solve_results)

        results[solver_name] = ComparisonResult(
            solver_name=solver_name,
            puzzles_solved=solved,
            total_puzzles=len(puzzles),
            solve_rate=solved / len(puzzles) if puzzles else 0.0,
            avg_time_ms=np.mean(times) if times else 0.0,
            median_time_ms=np.median(times) if times else 0.0,
            total_guesses=total_guesses,
            avg_guesses_per_puzzle=total_guesses / len(puzzles) if puzzles else 0.0,
            total_failed_guesses=total_failed,
            avg_failed_per_puzzle=total_failed / len(puzzles) if puzzles else 0.0,
        )

    return results


def print_comparison(results: Dict[str, ComparisonResult]) -> None:
    print("\n" + "=" * 90)
    print("SOLVER COMPARISON: Search-based vs Search-free")
    print("=" * 90)
    print()
    print("'Guesses' = search decisions where the solver tries a placement.")
    print("'Failed'  = guesses that led to backtracking (wrong guesses).")
    print()

    header = (
        f"{'Solver':<15} {'Solved':<10} {'Rate':<8} {'Avg Time':<12} "
        f"{'Guesses':<10} {'Failed':<10} {'Avg Guess':<10} {'Avg Fail':<10}"
    )
    print(header)
    print("-" * 90)

    for result in results.values():
        row = (
            f"{result.solver_name:<15} "
            f"{result.puzzles_solved}/{result.total_puzzles:<6} "
            f"{result.solve_rate:>5.1%}   "
            f"{result.avg_time_ms:>8.2f} ms  "
            f"{result.total_guesses:>8}  "
            f"{result.total_failed_guesses:>8}  "
            f"{result.avg_guesses_per_puzzle:>9.1f} "
            f"{result.avg_failed_per_puzzle:>9.1f}"
        )
        print(row)

    print("-" * 90)
    print()
    print("Key insight: Neural model makes 0 guesses - pure forward prediction.")
    print("All other solvers require search (guess + check + backtrack if wrong).")
    print("=" * 90)


if __name__ == "__main__":
    import json
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compare Queens solvers")
    parser.add_argument("--data", type=str, default="data/StateValSet.json")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.data) as f:
        all_puzzles = json.load(f)

    puzzles = [p for p in all_puzzles if p.get('step', 0) == 0][:args.limit]
    print(f"Loaded {len(puzzles)} puzzles")

    results = compare_solvers(
        puzzles,
        model_path=args.model,
        device=args.device,
        include_ortools=True,
    )

    print_comparison(results)
