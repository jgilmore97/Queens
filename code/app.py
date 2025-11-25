import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
from io import BytesIO
from PIL import Image

from improved_solver import Solver

# Configuration
MODEL_PATH = "checkpoints/transformer/HRM/best_model.pt"
PUZZLES_PATH = "puzzles.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global state
solver = None
current_puzzle = None
current_solution = None
current_activations = None
current_step = 0

def load_solver():
    global solver
    if solver is None:
        solver = Solver(MODEL_PATH, device=DEVICE)
    return solver

def load_puzzles():
    with open(PUZZLES_PATH, 'r') as f:
        return json.load(f)

def create_board_image(region_board, queen_board=None, size_px=400):
    """
    Create board visualization with optional queen placements.
    
    Args:
        region_board: n×n array of region IDs
        queen_board: n×n binary array of queen positions (optional)
        size_px: output image size in pixels
    """
    n = region_board.shape[0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Color the board by regions
    cmap = plt.get_cmap('tab20', np.max(region_board) + 1)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(-0.5, np.max(region_board) + 1.5),
        ncolors=np.max(region_board) + 1
    )
    ax.imshow(region_board, cmap=cmap, norm=norm)
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=2)
    
    # Place queens if provided
    if queen_board is not None:
        for row in range(n):
            for col in range(n):
                if queen_board[row][col] == 1:
                    ax.text(col, row, 'X', fontsize=24, ha='center', va='center',
                           color='black', fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, left=False, 
                  labelbottom=False, labelleft=False)
    ax.set_title('Queens Puzzle', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def create_activation_image(heatmap, title, size_px=400):
    """
    Create activation heatmap visualization.
    
    Args:
        heatmap: n×n array of activation values
        title: heatmap title (e.g., "L-Early")
        size_px: output image size in pixels
    """
    n = heatmap.shape[0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Heatmap with diverging colormap
    im = ax.imshow(heatmap, cmap='RdBu_r', vmin=0, vmax=1)
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both', bottom=False, left=False,
                  labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def initialize_puzzle(puzzle_name):
    """Load selected puzzle and reset state."""
    global current_puzzle, current_solution, current_activations, current_step
    
    puzzles = load_puzzles()
    puzzle_data = next(p for p in puzzles if p.get('name', p.get('source', '')) == puzzle_name)
    
    region = np.array(puzzle_data['region'])
    current_puzzle = {
        'region': region,
        'name': puzzle_data.get('name', puzzle_data.get('source', 'Unknown')),
        'size': region.shape[0]
    }
    current_solution = None
    current_activations = None
    current_step = 0
    
    # Show initial board
    board_img = create_board_image(current_puzzle['region'])
    
    status = f"Loaded {puzzle_name}. Click 'Start Solving' to begin."
    return board_img, status, gr.update(interactive=True), gr.update(interactive=False)

def start_solving(show_activations):
    """Run solver and prepare step-through data."""
    global current_solution, current_activations, current_step
    
    load_solver()
    
    # Solve with activation capture
    if show_activations:
        solution, activations = solver.solve_puzzle(
            current_puzzle,
            capture_activations=True,
            activation_metric='max_embedding'
        )
        current_activations = activations
    else:
        solution = solver.solve_puzzle(
            current_puzzle,
            capture_activations=False
        )
        current_activations = None
    
    current_solution = solution
    current_step = 0
    
    status = f"Solving {current_puzzle['name']}. Click 'Next Step' to advance."
    return status, gr.update(interactive=True), gr.update(interactive=False)

def next_step():
    """Advance to next queen placement, showing activations if enabled."""
    global current_step
    
    n = current_puzzle['size']
    
    if current_step >= n:
        return None, "Puzzle complete!", gr.update(interactive=False)
    
    images_to_show = []
    
    # Build partial board up to current step
    partial_board = np.zeros((n, n), dtype=int)
    queen_positions = np.argwhere(current_solution == 1)
    for i in range(current_step):
        r, c = queen_positions[i]
        partial_board[r, c] = 1
    
    # Show activation sequence if enabled
    if current_activations is not None:
        act_dict = current_activations[current_step]
        L_maps = act_dict['L']
        
        # Early
        img_early = create_activation_image(L_maps['early'], 'L-Early')
        images_to_show.append(img_early)
        
        # Mid
        img_mid = create_activation_image(L_maps['mid'], 'L-Mid')
        images_to_show.append(img_mid)
        
        # Late
        img_late = create_activation_image(L_maps['late'], 'L-Late')
        images_to_show.append(img_late)
    
    # Place the next queen
    r, c = queen_positions[current_step]
    partial_board[r, c] = 1
    
    # Final board with new queen
    board_img = create_board_image(current_puzzle['region'], partial_board)
    images_to_show.append(board_img)
    
    current_step += 1
    
    status = f"Placed queen {current_step}/{n} at ({r}, {c})"
    
    if current_step >= n:
        status += ". Puzzle complete!"
        return images_to_show, status, gr.update(interactive=False)
    
    return images_to_show, status, gr.update(interactive=True)

def reset_puzzle():
    """Reset to puzzle selection."""
    global current_puzzle, current_solution, current_activations, current_step
    current_puzzle = None
    current_solution = None
    current_activations = None
    current_step = 0
    
    return None, "Select a puzzle to begin.", gr.update(interactive=False), gr.update(interactive=False)

# Gradio Interface
with gr.Blocks(title="Queens Puzzle Solver") as demo:
    gr.Markdown("# Queens Puzzle Solver with HRM")
    gr.Markdown("Watch the Hierarchical Reasoning Model solve expert-level Queens puzzles.")
    
    with gr.Row():
        with gr.Column(scale=1):
            puzzles = load_puzzles()
            puzzle_names = [p['name'] for p in puzzles]
            
            puzzle_dropdown = gr.Dropdown(
                choices=puzzle_names,
                label="Select Puzzle",
                value=puzzle_names[0] if puzzle_names else None
            )
            
            show_activations_toggle = gr.Checkbox(
                label="Show Activation Visualizations",
                value=True
            )
            
            load_button = gr.Button("Load Puzzle", variant="primary")
            solve_button = gr.Button("Start Solving", interactive=False)
            next_button = gr.Button("Next Step", interactive=False)
            reset_button = gr.Button("Reset")
            
            status_text = gr.Textbox(
                label="Status",
                value="Select a puzzle to begin.",
                interactive=False
            )
        
        with gr.Column(scale=2):
            display_gallery = gr.Gallery(
                label="Visualization",
                show_label=False,
                elem_id="gallery",
                columns=1,
                rows=1,
                height=500,
                object_fit="contain"
            )
    
    # Event handlers
    load_button.click(
        fn=initialize_puzzle,
        inputs=[puzzle_dropdown],
        outputs=[display_gallery, status_text, solve_button, next_button]
    )
    
    solve_button.click(
        fn=start_solving,
        inputs=[show_activations_toggle],
        outputs=[status_text, next_button, solve_button]
    )
    
    next_button.click(
        fn=next_step,
        inputs=[],
        outputs=[display_gallery, status_text, next_button]
    )
    
    reset_button.click(
        fn=reset_puzzle,
        inputs=[],
        outputs=[display_gallery, status_text, solve_button, next_button]
    )

if __name__ == "__main__":
    demo.launch()