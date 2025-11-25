import cv2
import numpy as np
import argparse
import json

from PIL import Image
from sklearn.cluster import KMeans
import random

from collections import deque, Counter, defaultdict
import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
import time
from tqdm import tqdm

def detect_grid_size_by_black_lines(
    image_path, axis='horizontal', row_or_col=100, black_threshold=50, visualize=False
):
    """Detect grid size by counting transitions from non-black to black pixels along a scan line."""
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    if axis == 'horizontal':
        line = img_np[row_or_col, :, :]
    elif axis == 'vertical':
        line = img_np[:, row_or_col, :]
    else:
        raise ValueError("Axis must be 'horizontal' or 'vertical'")

    def is_black(pixel):
        return all(channel < black_threshold for channel in pixel)

    transitions = 0
    prev_black = is_black(line[0])
    change_indices = []

    for i, pixel in enumerate(line[1:], start=1):
        curr_black = is_black(pixel)
        if curr_black and not prev_black:
            transitions += 1
            change_indices.append(i)
        prev_black = curr_black

    if visualize:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].imshow(img_np)
        if axis == 'horizontal':
            axs[0].axhline(y=row_or_col, color='red', linestyle='--')
        else:
            axs[0].axvline(x=row_or_col, color='red', linestyle='--')
        axs[0].set_title("Scan line overlay")

        r, g, b = line[:, 0], line[:, 1], line[:, 2]
        axs[1].plot(r, label="R", color='r')
        axs[1].plot(g, label="G", color='g')
        axs[1].plot(b, label="B", color='b')
        for idx in change_indices:
            axs[1].axvline(x=idx, color='black', linestyle=':', alpha=0.5)
        axs[1].set_title("RGB values with non-black → black transitions")
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    return transitions + 1


def extract_rgb_tensor(image_path, row_scan=5, col_scan=5, center_ratio=0.4):
    """Extract RGB tensor from board image by sampling center region of each cell."""
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    grid_rows = detect_grid_size_by_black_lines(image_path, 'horizontal', row_or_col=row_scan)
    grid_cols = detect_grid_size_by_black_lines(image_path, 'vertical', row_or_col=col_scan)
    grid_size = min(grid_rows, grid_cols)

    height, width = img_np.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size

    rgb_tensor = np.zeros((grid_size, grid_size, 3), dtype=np.float32)  # [n, n, 3]

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width

            margin_y = int(cell_height * (1 - center_ratio) / 2)
            margin_x = int(cell_width * (1 - center_ratio) / 2)

            cy1, cy2 = y1 + margin_y, y2 - margin_y
            cx1, cx2 = x1 + margin_x, x2 - margin_x

            region = img_np[cy1:cy2, cx1:cx2]
            rgb_tensor[row, col] = region.reshape(-1, 3).mean(axis=0)

    return rgb_tensor

def quantize_rgb_tensor(rgb_tensor, threshold=30):
    """Convert RGB tensor to integer labels by clustering similar colors."""
    color_map = {}
    next_label = 0
    n = rgb_tensor.shape[0]
    int_tensor = np.zeros((n, n), dtype=int)  # [n, n]

    for row in range(n):
        for col in range(n):
            color = rgb_tensor[row, col]
            match = None
            for known_rgb, label in color_map.items():
                if np.linalg.norm(color - np.array(known_rgb)) < threshold:
                    match = label
                    break
            if match is None:
                match = next_label
                color_map[tuple(color)] = next_label
                next_label += 1
            int_tensor[row, col] = match

    return int_tensor

def extract(filepath, threshold=30):
    """Extract region tensor from a board image file."""
    board_rgb = extract_rgb_tensor(filepath)
    region_tensor = quantize_rgb_tensor(board_rgb, threshold=threshold)
    return region_tensor


def solve_queens(region):
    """Solve the queens puzzle using backtracking. Returns (positions, board) or (None, None) if unsolvable."""
    region = np.asarray(region)  # [n, n]
    n, m = region.shape
    assert n == m, "Board must be square"

    columns_used = set()
    regions_used = set()
    positions = []

    def backtrack(row):
        if row == n:
            return True
        for col in range(n):
            reg_id = region[row, col]
            # One queen per column and per region
            if col in columns_used or reg_id in regions_used:
                continue
            # No diagonal adjacency
            if any(abs(r - row) == 1 and abs(c - col) == 1 for r, c in positions):
                continue
            columns_used.add(col)
            regions_used.add(reg_id)
            positions.append((row, col))
            if backtrack(row + 1):
                return True
            positions.pop()
            columns_used.remove(col)
            regions_used.remove(reg_id)
        return False

    if backtrack(0):
        board = np.zeros((n, n), dtype=int)  # [n, n]
        for r, c in positions:
            board[r, c] = 1
        return positions, board
    else:
        return None, None

# Mutation functions for generating new training boards by growing/shrinking regions.
# They ensure resulting boards have contiguous regions and unique solutions.

def get_neighbors(x, y, shape):
    """Return orthogonal neighbors within bounds."""
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
            yield nx, ny

def is_contiguous(board, region_id):
    """Check if all cells of a region form one connected component via BFS."""
    visited = set()
    coords = np.argwhere(board == region_id)
    if len(coords) == 0:
        return False

    start = tuple(coords[0])
    queue = deque([start])
    visited.add(start)

    while queue:
        x, y = queue.popleft()
        for nx, ny in get_neighbors(x, y, board.shape):
            if (nx, ny) not in visited and board[nx, ny] == region_id:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return len(visited) == len(coords)

def mutate_region_frontier(board, max_attempts=100):
    """Mutate a region by growing or shrinking along the frontier while maintaining contiguity."""
    board = board.copy()
    regions = np.unique(board)

    for _ in range(max_attempts):
        region = random.choice(regions)
        region_mask = (board == region)
        frontier = []

        for x, y in zip(*np.where(region_mask)):
            for nx, ny in get_neighbors(x, y, board.shape):
                if board[nx, ny] != region:
                    frontier.append((x, y))
                    break

        if not frontier:
            continue

        action = random.choice(["grow", "shrink"])

        if action == "grow":
            x, y = random.choice(frontier)
            neighbors = list(get_neighbors(x, y, board.shape))
            random.shuffle(neighbors)

            for nx, ny in neighbors:
                neighbor_region = board[nx, ny]
                if neighbor_region != region:
                    new_board = board.copy()
                    new_board[nx, ny] = region

                    if is_contiguous(new_board, neighbor_region) and is_contiguous(new_board, region):
                        return new_board
        elif action == "shrink":
            x, y = random.choice(frontier)
            neighbors = list(get_neighbors(x, y, board.shape))
            random.shuffle(neighbors)

            for nx, ny in neighbors:
                neighbor_region = board[nx, ny]
                if neighbor_region != region:
                    new_board = board.copy()
                    new_board[x, y] = neighbor_region

                    if is_contiguous(new_board, region) and is_contiguous(new_board, neighbor_region):
                        return new_board

    return board

def count_solutions(region, max_solutions=None):
    """Count valid queen solutions for a board, optionally stopping early at max_solutions."""
    region = np.asarray(region)  # [n, n]
    n, m = region.shape
    assert n == m, "Board must be square"

    columns_used = set()
    regions_used = set()
    positions = []
    solution_count = 0

    def backtrack(row):
        nonlocal solution_count
        if row == n:
            solution_count += 1
            return

        for col in range(n):
            reg_id = region[row, col]
            if col in columns_used or reg_id in regions_used:
                continue
            if any(abs(r - row) == 1 and abs(c - col) == 1 for r, c in positions):
                continue

            columns_used.add(col)
            regions_used.add(reg_id)
            positions.append((row, col))

            backtrack(row + 1)

            positions.pop()
            columns_used.remove(col)
            regions_used.remove(reg_id)

            if max_solutions is not None and solution_count >= max_solutions:
                return

    backtrack(0)
    return solution_count

def generate_unique_mutated_board(original_board,
                                   min_percent=0.18,
                                   max_percent=0.45,
                                   max_solution_count=1,
                                   max_attempts=60,
                                   max_mutation_tries=1000):
    """Mutate board until target percentage of cells change and solution count is within limit."""
    n = original_board.shape[0]
    total_cells = n * n
    target_percent = random.uniform(min_percent, max_percent)
    target_changed_cells = int(total_cells * target_percent)

    for attempt in range(max_attempts):
        board = original_board.copy()
        changed = np.zeros_like(board, dtype=bool)
        mutation_attempts = 0

        while np.count_nonzero(changed) < target_changed_cells and mutation_attempts < max_mutation_tries:
            new_board = mutate_region_frontier(board)
            if not np.array_equal(new_board, board):
                diff = new_board != board
                changed |= diff
                board = new_board
            mutation_attempts += 1

        if np.count_nonzero(changed) >= target_changed_cells:
            solutions = count_solutions(board, max_solutions=max_solution_count + 1)
            if 1 <= solutions <= max_solution_count:
                return board

    return None

def hash_board(region):
    return hash(region.tobytes())

def expand_board_dataset(seed_dataset, target_size=5000):
    """Expand seed dataset via region-mutation until target_size boards exist."""
    generated_dataset   = []
    offspring_counter   = Counter()
    fail_streak         = defaultdict(int)
    seen_hashes         = set()

    available_pool = deque()
    for seed in seed_dataset:
        seed_entry = {
            "region"     : seed["region"],
            "iteration"  : 0,
            "source"     : seed.get("filename", "unknown")
        }
        available_pool.append(seed_entry)
        seen_hashes.add(hash_board(seed["region"]))

    round_count = 0
    while len(generated_dataset) < target_size and available_pool:
        round_count += 1
        print(f"\n=== Round {round_count} ===")
        print(f"Generated so far: {len(generated_dataset)}")
        print(f"Attempting {len(available_pool)} mutations...")

        successes   = 0
        next_pool   = deque()

        for entry in tqdm(list(available_pool), desc=f"Mutating (Round {round_count})"):
            base_region   = entry["region"]
            root_source   = entry["source"]
            parent_iter   = entry["iteration"]

            try:
                new_region = generate_unique_mutated_board(base_region)
                if new_region is None:
                    board_key = hash_board(base_region)
                    fail_streak[board_key] += 1
                    if fail_streak[board_key] >= 5:
                        print(f"Board {root_source} (iter {parent_iter}) "
                              "has failed to mutate in 5 consecutive rounds.")
                    next_pool.append(entry)
                    continue

                region_hash = hash_board(new_region)
                if region_hash in seen_hashes:
                    board_key = hash_board(base_region)
                    fail_streak[board_key] += 1
                    next_pool.append(entry)
                    continue

                positions, label_board = solve_queens(new_region)

                child_board = {
                    "region"          : new_region,
                    "queen_positions" : positions,
                    "label_board"     : label_board,
                    "source"          : root_source,
                    "iteration"       : parent_iter + 1
                }

                generated_dataset.append(child_board)
                next_pool.append(child_board)
                seen_hashes.add(region_hash)
                offspring_counter[root_source] += 1
                successes += 1

                fail_streak.pop(hash_board(base_region), None)

            except Exception as e:
                print(f"Error mutating {root_source} (iter {parent_iter}): {e}")
                next_pool.append(entry)

        print(f"Round {round_count} complete: {successes} new boards.")
        if successes == 0:
            print("No successful mutations this round — stopping early.")
            break

        available_pool = next_pool

    print(f"\n Finished: {len(generated_dataset)} boards generated.\n")
    print("Offspring per seed:")
    for seed, count in offspring_counter.items():
        print(f"  {seed}: {count}")

    return generated_dataset, offspring_counter

def visualize_queens_board(board: np.ndarray, title: str = "Colored Queens Board") -> None:
    """Visualize a board where each integer represents a distinct color region."""
    n = board.shape[0]
    assert board.shape[0] == board.shape[1], "Board must be square (n x n)."

    cmap = plt.get_cmap('tab20', np.max(board) + 1)
    norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(board) + 1.5), ncolors=np.max(board) + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(board, cmap=cmap, norm=norm, interpolation='none')

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    plt.show()

def save_stateless_dataset_json(dataset, save_path):
    """Save dataset to JSON with NumPy arrays converted to lists."""
    serializable_data = []

    for entry in dataset:
        serializable_data.append({
            "region": entry["region"].tolist(),
            "label_board": entry["label_board"].tolist(),
            "queen_positions": [list(pos) for pos in entry["queen_positions"]],
            "source": entry.get("source", "unknown")
        })

    with open(save_path, "w") as f:
        json.dump(serializable_data, f)

    print(f"Saved dataset to {save_path}")

def rotate_matrix_90(matrix):
    """Rotate a 2D list 90 degrees clockwise."""
    return [list(row) for row in zip(*matrix[::-1])]

def rotate_queen_positions(queen_positions, size, rotations):
    """Rotate queen positions by 90 degree increments."""
    positions = queen_positions
    for _ in range(rotations):
        positions = [[col, size - 1 - row] for row, col in positions]
    return positions

def rotate_board_data(board_data):
    """Generate 4 rotated versions (0, 90, 180, 270 degrees) of a queens board."""
    original_region = board_data['region']
    original_label_board = board_data['label_board']
    original_queen_positions = board_data['queen_positions']
    source = board_data.get('source', 'unknown_source.jpg')
    iteration = board_data.get('iteration', 0)
    size = len(original_region)

    rotated_boards = []

    region = original_region
    label_board = original_label_board
    queen_positions = original_queen_positions

    for i in range(4):
        rotated_boards.append({
            'region': copy.deepcopy(region),
            'label_board': copy.deepcopy(label_board),
            'queen_positions': copy.deepcopy(queen_positions),
            'source': f"{source}_rot{i*90}",
            'iteration': iteration
        })
        region = rotate_matrix_90(region)
        label_board = rotate_matrix_90(label_board)
        queen_positions = rotate_queen_positions(queen_positions, size, 1)

    return rotated_boards

def generate_training_states(board_data, state_0_only=False):
    """Generate training examples with progressive queen placements from a solved board."""
    region = board_data['region']
    full_solution = board_data['queen_positions']
    board_size = len(region)

    offset = random.randint(0, board_size - 1)
    rotated_queens = full_solution[offset:] + full_solution[:offset]

    training_examples = []
    current_input = [[0]*board_size for _ in range(board_size)]

    label_output = [[0]*board_size for _ in range(board_size)]
    for r, c in rotated_queens:
        label_output[r][c] = 1

    training_examples.append({
        'region': copy.deepcopy(region),
        'partial_board': copy.deepcopy(current_input),
        'label_board': label_output,
        'step': 0,
        'source': board_data.get('source', ''),
        'iteration': board_data.get('iteration', 0)
    })

    if state_0_only:
        return training_examples

    for i in range(board_size - 1):
        row, col = rotated_queens[i]
        current_input[row][col] = 1

        label_output = [[0]*board_size for _ in range(board_size)]
        for r, c in rotated_queens[i+1:]:
            label_output[r][c] = 1

        training_examples.append({
            'region': copy.deepcopy(region),
            'partial_board': copy.deepcopy(current_input),
            'label_board': label_output,
            'step': i + 1,
            'source': board_data.get('source', ''),
            'iteration': board_data.get('iteration', 0)
        })

    return training_examples

def save_state_dataset_to_json(dataset, filename="queens_training_data.json"):
    """Save training dataset to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} examples to {filename}")

def visualize_queens_board_with_queens(example: dict, title: str = "Queens Board with Queens", show_labels = False) -> None:
    """Visualize a training example with queens overlaid on the colored region board."""
    region_board = np.array(example['region'])  # [n, n]
    if not show_labels:
        queen_board = np.array(example['partial_board'])  # [n, n]
    else:
        queen_board = np.array(example['label_board'])  # [n, n]
    n = region_board.shape[0]

    assert region_board.shape == queen_board.shape, "Board shape mismatch."

    cmap = plt.get_cmap('tab20', np.max(region_board) + 1)
    norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(region_board) + 1.5), ncolors=np.max(region_board) + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(region_board, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)

    for row in range(n):
        for col in range(n):
            if queen_board[row][col] == 1:
                ax.text(col, row, "X", va='center', ha='center',
                        fontsize=24, color='black', fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    plt.show()
