import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from model import HRM, HeteroGAT, GAT, HRM_FullSpatial
from data_loader import build_heterogeneous_edge_index
from board_manipulation import solve_queens


class Solver:
    def __init__(self, model_path, device: str = 'cuda'):
        self.device = device
        self.model, self.is_heterogeneous = self.load_model(model_path)
        self.model = self.model.to(device)
        self.max_regions = 11
        self.model.eval()

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['config_dict']
        state_dict = checkpoint['model_state_dict']
    
        has_h_block = any(k.startswith('h_block.') for k in state_dict.keys())
        has_z_H_init = 'z_H_init' in state_dict
        is_hrm_spatial = has_h_block and has_z_H_init
        is_hrm = model_config.get('model_type') == 'HRM' or 'n_cycles' in model_config

        is_homogeneous_gat = ('layer_count' in model_config and
                              'hgt_heads' not in model_config and
                              not is_hrm)
        if is_hrm_spatial:
            model = HRM_FullSpatial(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                gat_heads=model_config.get('gat_heads', 2),
                hgt_heads=model_config.get('hgt_heads', 4),
                hmod_heads=model_config.get('hmod_heads', 4),
                dropout=model_config.get('dropout', 0.1),
                n_cycles=model_config.get('n_cycles', 3),
                t_micro=model_config.get('t_micro', 2)
            )
            print(f"Loaded HRM Full Spatial solver (cycles={model_config.get('n_cycles', 2)}, t_micro={model_config.get('t_micro', 2)})")
            is_heterogeneous = True

        elif is_hrm:
            model = HRM(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                gat_heads=model_config.get('gat_heads', 2),
                hgt_heads=model_config.get('hgt_heads', 4),
                dropout=model_config.get('dropout', 0.2),
                use_batch_norm=model_config.get('use_batch_norm', False),
                n_cycles=model_config.get('n_cycles', 3),
                t_micro=model_config.get('t_micro', 2),
                use_input_injection=model_config.get('use_input_injection', True),
                z_dim=model_config.get('z_dim', 128),
                use_hmod=model_config.get('use_hmod', False),
                same_size_batches=model_config.get('same_size_batches', False)
            )
            print(f"Loaded HRM solver (cycles={model_config.get('n_cycles', 2)}, t_micro={model_config.get('t_micro', 2)})")
            is_heterogeneous = True
            
        elif is_homogeneous_gat:
            model = GAT(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                layer_count=model_config['layer_count'],
                dropout=model_config['dropout'],
                heads=model_config.get('gat_heads', 2)
            )
            print(f"Loaded Homogeneous GAT solver")
            is_heterogeneous = False
        else:
            model = HeteroGAT(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                layer_count=model_config['layer_count'],
                dropout=model_config['dropout'],
                gat_heads=model_config['gat_heads'],
                hgt_heads=model_config['hgt_heads']
            )
            print(f"Loaded HeteroGAT solver")
            is_heterogeneous = True

        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from checkpoint")
        print(f"Model config: {model_config}")

        return model, is_heterogeneous

    def _build_node_features(self, region_board: np.ndarray, queen_board: np.ndarray) -> torch.Tensor:
        """Build node feature vectors combining normalized coordinates, one-hot region encoding, and queen flags."""
        n = region_board.shape[0]
        N2 = n * n

        coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)

        reg_onehot = np.zeros((N2, self.max_regions), dtype=np.float32)
        flat_ids = region_board.flatten()
        reg_onehot[np.arange(N2), flat_ids] = 1.0

        has_queen = queen_board.flatten()[:, None].astype(np.float32)

        features = np.hstack([coords, reg_onehot, has_queen])

        return torch.from_numpy(features)

    def _process_intermediates(self, intermediates: dict, n: int,
                             activation_metric: str = 'l2_norm') -> dict:
        """Convert HRM intermediates (L-states) to per-cycle activation heatmaps grouped by early/mid/late stages."""
        L_states = intermediates['L_states']  # List of 6 tensors [C, d]

        L_early = self._compute_cycle_activation(L_states[0:2], n, activation_metric)  # Micro 0-1 of cycle 0
        L_mid = self._compute_cycle_activation(L_states[2:4], n, activation_metric)    # Micro 0-1 of cycle 1
        L_late = self._compute_cycle_activation(L_states[4:6], n, activation_metric)   # Micro 0-1 of cycle 2

        return {
            'L': {
                'early': L_early,
                'mid': L_mid,
                'late': L_late
            }
        }

    def _compute_cycle_activation(self, state_list: List[torch.Tensor], n: int,
                                 activation_metric: str = 'l2_norm') -> np.ndarray:
        """Compute per-cell activation from L-states, averaged across timesteps and reshaped to [n, n]."""
        stacked = torch.stack(state_list, dim=0)  # [T, C, d]
        mean_state = stacked.mean(dim=0)  # [C, d]

        if activation_metric == 'l2_norm':
            activations = torch.norm(mean_state, dim=1)  # [C]
        elif activation_metric == 'mean_embedding':
            activations = mean_state.mean(dim=1)  # [C]
        elif activation_metric == 'max_embedding':
            activations = torch.max(torch.abs(mean_state), dim=1)[0]  # [C]
        else:
            raise ValueError(f"Unknown activation_metric: {activation_metric}")

        heatmap = activations.cpu().numpy().reshape(n, n)  # [n, n]

        if activation_metric == 'mean_embedding':
            # Use centered normalization for diverging around 0
            heatmap_abs_max = np.max(np.abs(heatmap))
            if heatmap_abs_max > 1e-6:
                heatmap = heatmap / heatmap_abs_max
        else:
            # Normalize to [0, 1] for L2 and max
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return heatmap

    def place_queen(self, region_board: np.ndarray, partial_board: np.ndarray,
                    edge_index_dict: Dict[str, torch.Tensor],
                    capture_activations: bool = False,
                    activation_metric: str = 'l2_norm') -> Tuple:
        """Place a queen at the highest-scoring cell using model predictions."""
        n = region_board.shape[0]

        node_features = self._build_node_features(region_board, partial_board)
        node_features = node_features.to(self.device)

        with torch.no_grad():
            x_dict = {'cell': node_features}
            edge_index_dict_formatted = {
                ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
            }

            if capture_activations and isinstance(self.model, HRM):
                logits, intermediates = self.model(x_dict, edge_index_dict_formatted,
                                                   return_intermediates=True)
                activation_dict = self._process_intermediates(intermediates, n, activation_metric)
            else:
                logits = self.model(x_dict, edge_index_dict_formatted)
                activation_dict = None

        logits_np = logits.cpu().numpy().reshape(n, n)  # [n, n]
        flat_logits = logits_np.flatten()
        top_idx = np.argmax(flat_logits)
        top_logit = flat_logits[top_idx]
        top_row, top_col = top_idx // n, top_idx % n

        if capture_activations and activation_dict is not None:
            activation_dict['placement'] = (top_row, top_col)
            activation_dict['logit'] = float(top_logit)
            return top_row, top_col, top_logit, activation_dict

        return top_row, top_col, top_logit

    def solve_puzzle(self, puzzle: dict, capture_activations: bool = False,
                    activation_metric: str = 'l2_norm') -> Tuple:
        """Solve a Queens puzzle autoregressively by placing n queens sequentially."""
        region_board = np.array(puzzle['region'])  # [n, n]
        n = region_board.shape[0]
        queen_board = np.zeros((n, n), dtype=int)  # [n, n]
        edge_index_dict = self._build_edge_index(region_board)
        activations = [] if capture_activations else None

        for step in range(n):
            if capture_activations:
                row, col, top_logit, act_dict = self.place_queen(
                    region_board, queen_board, edge_index_dict,
                    capture_activations=True,
                    activation_metric=activation_metric
                )
                activations.append(act_dict)
            else:
                row, col, top_logit = self.place_queen(
                    region_board, queen_board, edge_index_dict,
                    capture_activations=False
                )

            print(f"Placing queen at: ({row}, {col}) with logit score: {top_logit:.3f}")
            queen_board[row, col] = 1

        if capture_activations:
            return queen_board, activations
        return queen_board

    @staticmethod
    def solve_with_vanilla_backtracking(puzzle: dict):
        region_board = np.array(puzzle['region'])
        positions, board = solve_queens(region_board)
        return board

    def _build_edge_index(self, region_board: np.ndarray) -> Dict[str, torch.Tensor]:
        """Build heterogeneous edge index with line, region, and diagonal constraint edges."""
        edge_index_dict = build_heterogeneous_edge_index(region_board)

        for edge_type, edge_index in edge_index_dict.items():
            edge_index_dict[edge_type] = edge_index.to(self.device)

        return edge_index_dict


    def visualize_solution(self, puzzle: dict, solution: np.ndarray,
                          activations: List[dict], output_dir: Optional[str] = None,
                          show_regions: bool = True, activation_metric: str = 'max_embedding') -> None:
        """Visualize the reasoning progression across queen placements with L-state activation heatmaps."""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        region_board = np.array(puzzle['region'])  # [n, n]
        n = region_board.shape[0]
        num_placements = len(activations)

        placed_queens = []

        for step_idx, act_dict in enumerate(activations):
            row, col = act_dict['placement']
            logit = act_dict['logit']

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"Queen {step_idx + 1}/{num_placements}: Placement ({row}, {col}) | Logit: {logit:.3f}",
                        fontsize=14, fontweight='bold')

            ax = axes[0]
            if show_regions:
                self._draw_colored_board(ax, region_board, placed_queens, row, col)

            ax.set_title('Board State', fontsize=12, fontweight='bold')

            L_maps = act_dict['L']
            for col_idx, (stage_name, heatmap) in enumerate([
                ('Early', L_maps['early']),
                ('Mid', L_maps['mid']),
                ('Late', L_maps['late'])
            ]):
                ax = axes[col_idx + 1]

                im = ax.imshow(heatmap, cmap='RdBu_r', vmin=0, vmax=1)

                ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                ax.grid(which='minor', color='black', linewidth=1)

                ax.set_title(f"L-{stage_name}", fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(which='both', bottom=False, left=False,
                             labelbottom=False, labelleft=False)

                for prev_row, prev_col in placed_queens:
                    ax.text(prev_col, prev_row, 'X', fontsize=24, ha='center', va='center',
                           color='black', fontweight='bold')

                ax.scatter([col], [row], marker='*', color='lime', s=1000,
                          edgecolors='white', linewidths=3, zorder=10)

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                filename = output_path / f"step_{step_idx:02d}_queen_{row}_{col}.png"
                plt.savefig(filename, dpi=100, bbox_inches='tight')
                print(f"Saved: {filename}")
                plt.close()
            else:
                plt.show()

            placed_queens.append((row, col))

        if output_dir:
            self._create_summary_image(output_dir, num_placements, n)

    def _create_summary_image(self, output_dir, num_placements, n):
        """Concatenate all step visualizations vertically into a summary image."""
        from PIL import Image

        output_path = Path(output_dir)
        step_images = []

        for step_idx in range(num_placements):
            filename = output_path / f"step_{step_idx:02d}_queen_*.png"
            import glob
            matches = glob.glob(str(filename))
            if matches:
                img = Image.open(matches[0])
                step_images.append(img)

        if not step_images:
            print("No step images found for summary")
            return

        total_height = sum(img.height for img in step_images)
        max_width = step_images[0].width

        summary = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in step_images:
            summary.paste(img, (0, y_offset))
            y_offset += img.height

        summary_path = output_path / 'summary_all_steps.png'
        summary.save(summary_path)
        print(f"Saved summary: {summary_path}")

    def _draw_colored_board(self, ax, region_board, placed_queens, current_row, current_col):
        """Draw a colored board with regions, previous queen placements, and current placement marker."""
        n = region_board.shape[0]

        cmap = plt.get_cmap('tab20', np.max(region_board) + 1)
        norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(region_board) + 1.5),
                                   ncolors=np.max(region_board) + 1)
        ax.imshow(region_board, cmap=cmap, norm=norm)

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1)

        for prev_row, prev_col in placed_queens:
            ax.text(prev_col, prev_row, "X", va='center', ha='center',
                   fontsize=24, color='black', fontweight='bold')

        ax.scatter([current_col], [current_row], marker='*', color='lime', s=1000,
                  edgecolors='white', linewidths=3, zorder=10)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which='both', bottom=False, left=False,
                      labelbottom=False, labelleft=False)

    def _create_region_overlay(self, region_board: np.ndarray, n: int) -> np.ndarray:
        """Create a normalized region overlay for visualization colormapping."""
        overlay = region_board.astype(np.float32)
        overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
        return overlay

    def evaluate_solver(self, puzzle: dict) -> bool:
        model_solution = self.solve_puzzle(puzzle)
        backtrack_solution = self.solve_with_vanilla_backtracking(puzzle)
        return np.array_equal(model_solution, backtrack_solution)
