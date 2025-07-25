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
from matplotlib import colors

import os
import time
from tqdm import tqdm

# function to detect grid size by counting transitions from non-black to black pixels along a line of pixels (counts black lines to detect grid size)
def detect_grid_size_by_black_lines(
    image_path, axis='horizontal', row_or_col=100, black_threshold=50, visualize=False
):

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


#function to turn a jpg image of queens board into a tensor of RGB values
def extract_rgb_tensor(image_path, row_scan=5, col_scan=5, center_ratio=0.4):
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    grid_rows = detect_grid_size_by_black_lines(image_path, 'horizontal', row_or_col=row_scan)
    grid_cols = detect_grid_size_by_black_lines(image_path, 'vertical', row_or_col=col_scan)
    grid_size = min(grid_rows, grid_cols)

    height, width = img_np.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size

    rgb_tensor = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

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

# combine with extract_rgb_tensor to create a function that turns the RGB tensor into integer labels ->every color in the RGB tensor is assigned a unique integer label
def quantize_rgb_tensor(rgb_tensor, threshold=30):
    """
    Converts RGB tensor to integer-labeled tensor using per-image color clustering.

    Args:
        rgb_tensor (np.ndarray): n x n x 3 RGB tensor.
        threshold (float): Distance threshold for matching colors.

    Returns:
        np.ndarray: n x n integer tensor.
    """
    color_map = {}
    next_label = 0
    n = rgb_tensor.shape[0]
    int_tensor = np.zeros((n, n), dtype=int)

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

# combine above 2 functions into one to fully
def extract(filepath, threshold=30):
    board_rgb = extract_rgb_tensor(filepath)
    region_tensor = quantize_rgb_tensor(board_rgb, threshold=threshold)
    return region_tensor


# uses traditional backtracking to solve the queens puzzle. Using to get label data.
def solve_queens(region):
    """
    Solve the 'queens' puzzle on a square grid whose regions are encoded
    by integer IDs in `region` (an N×N array).
    Returns (positions, board):
      - positions: list of (row,col) if a solution exists, otherwise None
      - board: N×N array with 1 at queen positions and 0 elsewhere, or None
    """
    region = np.asarray(region)
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
            # 1) one queen per column
            if col in columns_used or reg_id in regions_used:
                continue
            # 2) no diagonal adjacency
            if any(abs(r - row) == 1 and abs(c - col) == 1 for r, c in positions):
                continue
            # place
            columns_used.add(col)
            regions_used.add(reg_id)
            positions.append((row, col))
            if backtrack(row + 1):
                return True
            # undo
            positions.pop()
            columns_used.remove(col)
            regions_used.remove(reg_id)
        return False

    if backtrack(0):
        # build the board array
        board = np.zeros((n, n), dtype=int)
        for r, c in positions:
            board[r, c] = 1
        return positions, board
    else:
        return None, None

# ---------------------------------------------------------------------------------------------------------------------------------------------
'''
below bounded functions are used together to generate new training data boards by mutating existing ones.
they mutate the board by growing or shrinking regions and checking if the resulting board is still valid, 
ie all regions are contiguous, puzzle is still solvable, and there is only one solution.
'''

def get_neighbors(x, y, shape):
    """Return orthogonal neighbors."""
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
            yield nx, ny

def is_contiguous(board, region_id):
    """Check if all cells of a region form one connected component."""
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
    """Mutate a region by growing/shrinking along the frontier."""
    board = board.copy()
    regions = np.unique(board)

    for _ in range(max_attempts):
        region = random.choice(regions)
        region_mask = (board == region)
        frontier = []

        # Find frontier: cells adjacent to a different region
        for x, y in zip(*np.where(region_mask)):
            for nx, ny in get_neighbors(x, y, board.shape):
                if board[nx, ny] != region:
                    frontier.append((x, y))
                    break  # it's a border cell

        if not frontier:
            continue  # no valid frontier, try another region

        action = random.choice(["grow", "shrink"])

        if action == "grow":
            # Pick a border cell and try to steal a neighboring cell
            x, y = random.choice(frontier)
            neighbors = list(get_neighbors(x, y, board.shape))
            random.shuffle(neighbors)

            for nx, ny in neighbors:
                neighbor_region = board[nx, ny]
                if neighbor_region != region:
                    # Tentatively change cell to our region
                    new_board = board.copy()
                    new_board[nx, ny] = region

                    if is_contiguous(new_board, neighbor_region) and is_contiguous(new_board, region):
                        return new_board
        elif action == "shrink":
            # Try removing a border cell and assigning it to a neighbor
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

    return board  # no mutation found, return original

def count_solutions(region, max_solutions=None):
    """
    Counts the number of valid queen solutions for a board using the same logic
    as solve_queens(), but without stopping at the first solution.

    Parameters:
    - region: N×N numpy array of region IDs (colors)
    - max_solutions: Optional cap to stop early

    Returns:
    - Integer: number of unique valid solutions
    """
    region = np.asarray(region)
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
    """
    Mutates the board until a proportion of its cells are changed and it has a unique (or limited) solution count.
    Returns:
        A new board, or None if unable to create a valid one in the allotted attempts.
    """
    n = original_board.shape[0]
    total_cells = n * n
    target_percent = random.uniform(min_percent, max_percent)
    # print(target_percent)
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
    """
    Expand a seed dataset via region-mutation until `target_size` boards exist.

    Each returned board dict contains:
      'region', 'queen_positions', 'solution_board',
      'source'     -> original seed filename
      'iteration'  -> 0 for seed, 1,2,… for descendants
    """
    # ---- initialise bookkeeping ----
    generated_dataset   = []
    offspring_counter   = Counter()                 # how many boards per seed
    fail_streak         = defaultdict(int)          # consecutive failed rounds per board hash
    seen_hashes         = set()                     # every region we've accepted

    # prime the pool with SEEDS (iteration = 0)
    available_pool = deque()
    for seed in seed_dataset:
        seed_entry = {
            "region"     : seed["region"],
            "iteration"  : 0,
            "source"     : seed.get("filename", "unknown")
        }
        available_pool.append(seed_entry)
        seen_hashes.add(hash_board(seed["region"]))   # seeds are "seen" but not counted as generated

    round_count = 0
    # ------------- main loop -------------
    while len(generated_dataset) < target_size and available_pool:
        round_count += 1
        print(f"\n=== Round {round_count} ===")
        print(f"Generated so far: {len(generated_dataset)}")
        print(f"Attempting {len(available_pool)} mutations...")

        successes   = 0
        next_pool   = deque()

        for entry in tqdm(list(available_pool), desc=f"Mutating (Round {round_count})"):
            base_region   = entry["region"]
            root_source   = entry["source"]      # already the seed filename
            parent_iter   = entry["iteration"]

            try:
                new_region = generate_unique_mutated_board(base_region)
                if new_region is None:
                    # ➜ mutation failed; keep board, update fail-streak
                    board_key = hash_board(base_region)
                    fail_streak[board_key] += 1
                    if fail_streak[board_key] >= 5:
                        print(f"⚠️  Board {root_source} (iter {parent_iter}) "
                              "has failed to mutate in 5 consecutive rounds.")
                    next_pool.append(entry)
                    continue

                # ----- mutation succeeded -----
                region_hash = hash_board(new_region)
                if region_hash in seen_hashes:
                    # duplicate board, treat as failure for streak counting
                    board_key = hash_board(base_region)
                    fail_streak[board_key] += 1
                    next_pool.append(entry)
                    continue

                # unique & valid → solve to get queen positions
                positions, solution_board = solve_queens(new_region)

                # child board dict
                child_board = {
                    "region"          : new_region,
                    "queen_positions" : positions,
                    "solution_board"  : solution_board,
                    "source"          : root_source,          # propagate root seed filename
                    "iteration"       : parent_iter + 1
                }

                # update global structures
                generated_dataset.append(child_board)
                next_pool.append(child_board)
                seen_hashes.add(region_hash)
                offspring_counter[root_source] += 1
                successes += 1

                # reset fail-streak for the *parent* (it finally succeeded)
                fail_streak.pop(hash_board(base_region), None)

            except Exception as e:
                print(f"⚠️  Error mutating {root_source} (iter {parent_iter}): {e}")
                next_pool.append(entry)  # keep for another try

        print(f"Round {round_count} complete: {successes} new boards.")
        if successes == 0:
            print("No successful mutations this round — stopping early.")
            break

        available_pool = next_pool   # advance to next generation

    print(f"\n🎉 Finished: {len(generated_dataset)} boards generated.\n")
    print("Offspring per seed:")
    for seed, count in offspring_counter.items():
        print(f"  {seed}: {count}")

    return generated_dataset, offspring_counter

# def expand_board_dataset(seed_dataset, target_size=5000):
#     """
#     Expands a seed dataset of region boards to the target size using mutation and validation.
    
#     Parameters:
#     - seed_dataset: list of dicts, each with at least a 'region' and 'filename' field
#     - target_size: total number of generated boards desired
    
#     Returns:
#     - generated_dataset: list of valid, unique, mutated board dicts
#     """
#     generated_dataset = []
#     available_pool = deque(seed_dataset.copy())
#     seen_hashes = set(hash_board(entry['region']) for entry in seed_dataset)

#     round_count = 0

#     while len(generated_dataset) < target_size and available_pool:
#         round_count += 1
#         print(f"\n=== Round {round_count} ===")
#         print(f"Generated so far: {len(generated_dataset)}")
#         print(f"Attempting {len(available_pool)} mutations...")

#         successes = 0
#         next_pool = deque()

#         for entry in tqdm(list(available_pool), desc=f"Mutating (Round {round_count})"):
#             base_region = entry["region"]

#             try:
#                 new_region = generate_unique_mutated_board(base_region)
#                 if new_region is None:
#                     next_pool.append(entry)
#                     continue

#                 region_hash = hash_board(new_region)
#                 if region_hash in seen_hashes:
#                     continue

#                 positions, solution_board = solve_queens(new_region)

#                 new_board = {
#                     "region": new_region,
#                     "queen_positions": positions,
#                     "solution_board": solution_board,
#                     "source": entry.get("filename", "unknown"),
#                 }

#                 generated_dataset.append(new_board)
#                 next_pool.append(new_board)
#                 seen_hashes.add(region_hash)
#                 successes += 1

#             except Exception as e:
#                 print(f"⚠️ Error mutating {entry.get('filename', 'unknown')}: {e}")
#                 continue

#         print(f"Round {round_count} complete: {successes} new boards.")
#         if successes == 0:
#             print("No successful mutations this round — stopping early.")
#             break

#         available_pool = next_pool

#     print(f"\n Finished: {len(generated_dataset)} boards generated.")
#     return generated_dataset


# def generate_unique_mutated_board(original_board, target_mutations=6,
#                                    max_solution_count=1, max_attempts=30,
#                                    max_mutation_tries=100):
#     """
#     Tries to create a sufficiently mutated board with 1 to `max_solution_count` solutions.

#     Returns:
#     - A new board or None if no valid board was found.
#     """
#     for attempt in range(max_attempts):
#         board = original_board.copy()
#         successful_mutations = 0
#         total_mutation_attempts = 0

#         while successful_mutations < target_mutations and total_mutation_attempts < max_mutation_tries:
#             new_board = mutate_region_frontier(board)
#             if not np.array_equal(new_board, board):
#                 board = new_board
#                 successful_mutations += 1
#             total_mutation_attempts += 1

#         # must have at least 1 and at most max_solution_count solutions
#         solutions = count_solutions(board, max_solutions=max_solution_count + 1)
#         if 1 <= solutions <= max_solution_count:
#             return board

#     return None
# ---------------------------------------------------------------------------------------------------------------------------------------------

# take a board tensor and visualize it as a colored grid -> recreating the starting viz
def visualize_queens_board(board: np.ndarray, title: str = "Colored Queens Board") -> None:
    """
    Visualize a Queens board where each integer represents a distinct color region.

    Args:
        board (np.ndarray): 2D numpy array with integer values representing color regions.
        title (str): Title for the plot.
    """
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
    """
    Saves the dataset to a JSON file.
    All NumPy arrays are converted to native Python lists.
    """
    serializable_data = []

    for entry in dataset:
        serializable_data.append({
            "region": entry["region"].tolist(),
            "solution_board": entry["solution_board"].tolist(),
            "queen_positions": [list(pos) for pos in entry["queen_positions"]],
            "source": entry.get("source", "unknown")
        })

    with open(save_path, "w") as f:
        json.dump(serializable_data, f)

    print(f"✅ Saved dataset to {save_path}")

#-----------------------------------------------------------------------------------------------------
# Below functions perform data augmentation by rotating the board and queen positions.
# They generate 4 rotated versions (0°, 90°, 180°, 270°
def rotate_matrix_90(matrix):
    """Rotate a 2D list 90 degrees clockwise."""
    return [list(row) for row in zip(*matrix[::-1])]

def rotate_queen_positions(queen_positions, size, rotations):
    """Rotate queen positions by 90° increments."""
    positions = queen_positions
    for _ in range(rotations):
        positions = [[col, size - 1 - row] for row, col in positions]
    return positions

def rotate_board_data(board_data):
    """
    Generate 4 rotated versions (0°, 90°, 180°, 270°) of a queens board.

    Input:
    - board_data: dict with 'region', 'queen_positions', 'solution_board', 'source'

    Output:
    - List of 4 dicts, each rotated version with keys: region, queen_positions, solution_board, source
    """
    original_region = board_data['region']
    original_solution_board = board_data['solution_board']
    original_queen_positions = board_data['queen_positions']
    source = board_data.get('source', 'unknown_source.jpg')
    iteration = board_data.get('iteration', 0)
    size = len(original_region)

    rotated_boards = []

    region = original_region
    solution_board = original_solution_board
    queen_positions = original_queen_positions

    for i in range(4):
        rotated_boards.append({
            'region': copy.deepcopy(region),
            'solution_board': copy.deepcopy(solution_board),
            'queen_positions': copy.deepcopy(queen_positions),
            'source': f"{source}_rot{i*90}",
            'iteration': iteration
        })
        # Prepare for next rotation
        region = rotate_matrix_90(region)
        solution_board = rotate_matrix_90(solution_board)
        queen_positions = rotate_queen_positions(queen_positions, size, 1)

    return rotated_boards
#-----------------------------------------------------------------------------------------------------

def generate_training_states(board_data):
    region = board_data['region']
    full_solution = board_data['queen_positions']
    board_size = len(region)

    offset = random.randint(0, board_size - 1)
    rotated_queens = full_solution[offset:] + full_solution[:offset]

    training_examples = []
    current_input = [[0]*board_size for _ in range(board_size)]

    # ✅ Step 0: No queens placed yet
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

    # Steps 1 to N-1
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


# Below function saves the dataset after state creation to a JSON file.
def save_state_dataset_to_json(dataset, filename="queens_training_data.json"):
    """
    Saves the dataset to a JSON file in the current directory.

    Args:
        dataset (list): List of training examples (dicts).
        filename (str): Name of the output JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} examples to {filename}")


# Example usage:
# save_dataset_to_json(all_training_data, "augmented_queens_data.json")

# Below function visualizes a single training example with queens overlaid on the board.
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def visualize_queens_board_with_queens(example: dict, title: str = "Queens Board with Queens") -> None:
    """
    Visualize a single training example from the dataset with queens overlaid.

    Args:
        example (dict): A single example from your dataset containing 'region' and 'partial_board'.
        title (str): Optional title for the plot.
    """
    region_board = np.array(example['region'])
    queen_board = np.array(example['partial_board'])
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


# Example usage:
# visualize_queens_board_with_queens(all_training_data[0], title="Step 1 Example")

