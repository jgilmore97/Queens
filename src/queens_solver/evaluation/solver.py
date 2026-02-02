import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from queens_solver.models.models import HRM, HeteroGAT, GAT
from queens_solver.models.benchmark import BenchmarkHRM, BenchmarkSequential

logger = logging.getLogger(__name__)
from queens_solver.data.dataset import build_heterogeneous_edge_index
from queens_solver.data.preprocessing import solve_queens


class _SingleBatch:
    """Minimal batch wrapper for single-puzzle inference with HRM models."""
    __slots__ = ('x_dict', 'edge_index_dict', 'num_graphs', '_cell')

    def __init__(self, x: torch.Tensor, edge_dict: Dict, device):
        self.x_dict = {'cell': x}
        self.edge_index_dict = edge_dict
        self.num_graphs = 1
        self._cell = type('CellData', (), {
            'x': x,
            'batch': torch.zeros(x.shape[0], dtype=torch.long, device=device),
            'num_nodes': x.shape[0]
        })()

    def __getitem__(self, key):
        if key == 'cell':
            return self._cell
        return self.edge_index_dict.get(key)


class Solver:
    def __init__(self, model_path, device: str = 'cuda'):
        self.device = device
        self.is_benchmark = False
        self.model, self.is_heterogeneous = self.load_model(model_path)
        self.model = self.model.to(device)
        self.max_regions = 11
        self.model.eval()

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['config_dict']
        state_dict = checkpoint['model_state_dict']

        model_type = model_config.get('model_type', '')

        has_lowercase_l_block = any(k.startswith('l_block.') for k in state_dict.keys())
        has_lowercase_h_block = any(k.startswith('h_block.') for k in state_dict.keys())
        has_uppercase_L_block = any(k.startswith('L_block.') for k in state_dict.keys())
        has_uppercase_H_block = any(k.startswith('H_block.') for k in state_dict.keys())
        has_numbered_layers = any(k.startswith('layers.') for k in state_dict.keys())
        has_z_H_init = 'z_H_init' in state_dict
        has_gat_heads = 'gat_heads' in model_config

        is_benchmark_sequential = (model_type == 'benchmark_sequential' or
                                   (has_numbered_layers and not has_uppercase_L_block and not has_uppercase_H_block) or
                                   ('layers' in model_config and not has_gat_heads and 'n_cycles' not in model_config))

        is_benchmark_hrm = (model_type == 'benchmark_hrm' or
                           has_uppercase_L_block or has_uppercase_H_block or
                           ('n_cycles' in model_config and not has_gat_heads and not has_lowercase_l_block)) and not is_benchmark_sequential

        is_hrm_spatial = has_lowercase_h_block and has_z_H_init and has_lowercase_l_block
        is_hrm = (model_type == 'HRM' or 'n_cycles' in model_config) and has_gat_heads and not is_benchmark_hrm

        is_homogeneous_gat = ('layer_count' in model_config and
                              'hgt_heads' not in model_config and
                              not is_hrm and not is_hrm_spatial and
                              not is_benchmark_hrm and not is_benchmark_sequential)

        if is_benchmark_sequential:
            model = BenchmarkSequential(
                input_dim=model_config.get('input_dim', 14),
                hidden_dim=model_config.get('hidden_dim', 128),
                p_drop=model_config.get('dropout', 0.1),
                n_heads=model_config.get('n_heads', 4),
                layers=model_config.get('layers', 6)
            )
            logger.debug("Loaded BenchmarkSequential solver")
            is_heterogeneous = False
            self.is_benchmark = True

        elif is_benchmark_hrm:
            model = BenchmarkHRM(
                input_dim=model_config.get('input_dim', 14),
                hidden_dim=model_config.get('hidden_dim', 128),
                p_drop=model_config.get('dropout', 0.1),
                n_heads=model_config.get('n_heads', 4),
                n_cycles=model_config.get('n_cycles', 3),
                t_micro=model_config.get('t_micro', 2)
            )
            logger.debug(f"Loaded BenchmarkHRM solver (cycles={model_config.get('n_cycles', 3)}, t_micro={model_config.get('t_micro', 2)})")
            is_heterogeneous = False
            self.is_benchmark = True

        elif is_hrm_spatial:
            model = HRM(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                gat_heads=model_config.get('gat_heads', 2),
                hgt_heads=model_config.get('hgt_heads', 4),
                hmod_heads=model_config.get('hmod_heads', 4),
                dropout=model_config.get('dropout', 0.1),
                n_cycles=model_config.get('n_cycles', 3),
                t_micro=model_config.get('t_micro', 2)
            )
            logger.debug(f"Loaded HRM Full Spatial solver (cycles={model_config.get('n_cycles', 2)}, t_micro={model_config.get('t_micro', 2)})")
            is_heterogeneous = True
            self.is_benchmark = False

        elif is_hrm:
            model = HRM(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                gat_heads=model_config.get('gat_heads', 2),
                hgt_heads=model_config.get('hgt_heads', 4),
                hmod_heads=model_config.get('hmod_heads', 4),
                dropout=model_config.get('dropout', 0.2),
                n_cycles=model_config.get('n_cycles', 3),
                t_micro=model_config.get('t_micro', 2),
            )
            logger.debug(f"Loaded HRM solver (cycles={model_config.get('n_cycles', 2)}, t_micro={model_config.get('t_micro', 2)})")
            is_heterogeneous = True
            self.is_benchmark = False

        elif is_homogeneous_gat:
            model = GAT(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                layer_count=model_config['layer_count'],
                dropout=model_config['dropout'],
                heads=model_config.get('gat_heads', 2)
            )
            logger.debug("Loaded Homogeneous GAT solver")
            is_heterogeneous = False
            self.is_benchmark = False
        else:
            model = HeteroGAT(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                layer_count=model_config['layer_count'],
                dropout=model_config['dropout'],
                gat_heads=model_config['gat_heads'],
                hgt_heads=model_config['hgt_heads']
            )
            logger.debug("Loaded HeteroGAT solver")
            is_heterogeneous = True
            self.is_benchmark = False

        model.load_state_dict(checkpoint['model_state_dict'])

        logger.debug(f"Model loaded from checkpoint")
        logger.debug(f"Model config: {model_config}")

        return model, is_heterogeneous

    def _pad_board(self, board: np.ndarray, target_size: int, pad_value: int) -> np.ndarray:
        """Pad a board to target_size x target_size."""
        n = board.shape[0]
        if n >= target_size:
            return board
        padded = np.full((target_size, target_size), pad_value, dtype=board.dtype)
        padded[:n, :n] = board
        return padded

    def _build_node_features(self, region_board: np.ndarray, queen_board: np.ndarray,
                              pad_for_benchmark: bool = False) -> torch.Tensor:
        """Build node feature vectors combining normalized coordinates, one-hot region encoding, and queen flags."""
        if pad_for_benchmark:
            region_board = self._pad_board(region_board, self.max_regions, pad_value=-1)
            queen_board = self._pad_board(queen_board, self.max_regions, pad_value=0)

        n = region_board.shape[0]
        N2 = n * n

        coords = np.indices((n, n)).reshape(2, -1).T.astype(np.float32) / (n - 1)

        reg_onehot = np.zeros((N2, self.max_regions), dtype=np.float32)
        flat_ids = region_board.flatten()
        valid_mask = flat_ids >= 0
        reg_onehot[valid_mask, flat_ids[valid_mask]] = 1.0

        has_queen = queen_board.flatten()[:, None].astype(np.float32)

        features = np.hstack([coords, reg_onehot, has_queen])

        return torch.from_numpy(features)

    def _process_intermediates(self, intermediates: dict, n: int,
                             activation_metric: str = 'l2_norm') -> dict:
        """Convert HRM intermediates to per-cycle activation heatmaps."""
        L_states = intermediates['L_states']
        H_states = intermediates.get('H_states', None)

        num_L = len(L_states)
        t_micro = 2
        n_cycles = num_L // t_micro

        result = {'L': {}}

        if n_cycles >= 3:
            result['L']['early'] = self._compute_cycle_activation(L_states[0:2], n, activation_metric)
            result['L']['mid'] = self._compute_cycle_activation(L_states[2:4], n, activation_metric)
            result['L']['late'] = self._compute_cycle_activation(L_states[4:6], n, activation_metric)
        elif n_cycles == 2:
            result['L']['early'] = self._compute_cycle_activation(L_states[0:2], n, activation_metric)
            result['L']['late'] = self._compute_cycle_activation(L_states[2:4], n, activation_metric)
        else:
            result['L']['early'] = self._compute_cycle_activation(L_states, n, activation_metric)

        if H_states is not None and len(H_states) > 0:
            result['H'] = {}
            if len(H_states) >= 3:
                result['H']['early'] = self._compute_single_activation(H_states[0], n, activation_metric)
                result['H']['mid'] = self._compute_single_activation(H_states[1], n, activation_metric)
                result['H']['late'] = self._compute_single_activation(H_states[2], n, activation_metric)
            elif len(H_states) == 2:
                result['H']['early'] = self._compute_single_activation(H_states[0], n, activation_metric)
                result['H']['late'] = self._compute_single_activation(H_states[1], n, activation_metric)
            else:
                result['H']['final'] = self._compute_single_activation(H_states[0], n, activation_metric)

        return result

    def _compute_single_activation(self, state: torch.Tensor, n: int,
                                    activation_metric: str = 'l2_norm') -> np.ndarray:
        """Compute per-cell activation from a single state tensor [N, d], reshaped to [n, n]."""
        if activation_metric == 'l2_norm':
            activations = torch.norm(state, dim=1)
        elif activation_metric == 'mean_embedding':
            activations = state.mean(dim=1)
        elif activation_metric == 'max_embedding':
            activations = torch.max(torch.abs(state), dim=1)[0]
        else:
            raise ValueError(f"Unknown activation_metric: {activation_metric}")

        heatmap = activations.cpu().numpy().reshape(n, n)

        if activation_metric == 'mean_embedding':
            heatmap_abs_max = np.max(np.abs(heatmap))
            if heatmap_abs_max > 1e-6:
                heatmap = heatmap / heatmap_abs_max
        else:
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return heatmap

    def _compute_cycle_activation(self, state_list: List[torch.Tensor], n: int,
                                 activation_metric: str = 'l2_norm') -> np.ndarray:
        """Compute per-cell activation from L-states, averaged across timesteps and reshaped to [n, n]."""
        stacked = torch.stack(state_list, dim=0)
        mean_state = stacked.mean(dim=0)

        if activation_metric == 'l2_norm':
            activations = torch.norm(mean_state, dim=1)
        elif activation_metric == 'mean_embedding':
            activations = mean_state.mean(dim=1)
        elif activation_metric == 'max_embedding':
            activations = torch.max(torch.abs(mean_state), dim=1)[0]
        else:
            raise ValueError(f"Unknown activation_metric: {activation_metric}")

        heatmap = activations.cpu().numpy().reshape(n, n)

        if activation_metric == 'mean_embedding':
            heatmap_abs_max = np.max(np.abs(heatmap))
            if heatmap_abs_max > 1e-6:
                heatmap = heatmap / heatmap_abs_max
        else:
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        return heatmap

    def place_queen(self, region_board: np.ndarray, partial_board: np.ndarray,
                    edge_index_dict: Dict[str, torch.Tensor],
                    capture_activations: bool = False,
                    activation_metric: str = 'l2_norm',
                    return_logits: bool = False) -> Tuple:
        """Place a queen at the highest-scoring cell using model predictions."""
        n = region_board.shape[0]
        activation_dict = None

        with torch.no_grad():
            if getattr(self, 'is_benchmark', False):
                node_features = self._build_node_features(region_board, partial_board, pad_for_benchmark=True)
                node_features = node_features.to(self.device)
                x_batched = node_features.unsqueeze(0)
                logits = self.model(x_batched)
                logits = logits.squeeze(0).squeeze(-1)
                logits_padded = logits.cpu().numpy().reshape(self.max_regions, self.max_regions)
                logits_np = logits_padded[:n, :n]

            elif isinstance(self.model, HRM):
                node_features = self._build_node_features(region_board, partial_board)
                node_features = node_features.to(self.device)
                edge_index_dict_formatted = {
                    ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                    ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                    ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
                }
                batch = _SingleBatch(node_features, edge_index_dict_formatted, self.device)
                if capture_activations:
                    logits, intermediates = self.model(batch, return_intermediates=True)
                    activation_dict = self._process_intermediates(intermediates, n, activation_metric)
                else:
                    logits = self.model(batch)
                logits_np = logits.cpu().numpy().reshape(n, n)

            elif isinstance(self.model, HeteroGAT):
                node_features = self._build_node_features(region_board, partial_board)
                node_features = node_features.to(self.device)
                edge_index_dict_formatted = {
                    ('cell', 'line_constraint', 'cell'): edge_index_dict['line_constraint'],
                    ('cell', 'region_constraint', 'cell'): edge_index_dict['region_constraint'],
                    ('cell', 'diagonal_constraint', 'cell'): edge_index_dict['diagonal_constraint'],
                }
                batch = _SingleBatch(node_features, edge_index_dict_formatted, self.device)
                logits = self.model(batch)
                logits_np = logits.cpu().numpy().reshape(n, n)

            elif isinstance(self.model, GAT):
                node_features = self._build_node_features(region_board, partial_board)
                node_features = node_features.to(self.device)
                edge_indices = [
                    edge_index_dict['line_constraint'],
                    edge_index_dict['region_constraint'],
                    edge_index_dict['diagonal_constraint'],
                ]
                combined_edge_index = torch.cat(edge_indices, dim=1)
                logits = self.model(node_features, combined_edge_index)
                logits_np = logits.cpu().numpy().reshape(n, n)

            else:
                raise ValueError(f"Unknown model type: {type(self.model)}")

        flat_logits = logits_np.flatten()
        top_idx = np.argmax(flat_logits)
        top_logit = flat_logits[top_idx]
        top_row, top_col = top_idx // n, top_idx % n

        if capture_activations and activation_dict is not None:
            activation_dict['placement'] = (top_row, top_col)
            activation_dict['logit'] = float(top_logit)
            if return_logits:
                return top_row, top_col, top_logit, activation_dict, logits_np
            return top_row, top_col, top_logit, activation_dict

        if return_logits:
            return top_row, top_col, top_logit, logits_np
        return top_row, top_col, top_logit

    def solve_puzzle(self, puzzle: dict, capture_activations: bool = False,
                    activation_metric: str = 'l2_norm') -> Tuple:
        """Solve a Queens puzzle autoregressively by placing n queens sequentially."""
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        queen_board = np.zeros((n, n), dtype=int)
        edge_index_dict = self._build_edge_index(region_board)
        activations = [] if capture_activations else None
        placement_order = []

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

            logger.debug(f"Placing queen at: ({row}, {col}) with logit score: {top_logit:.3f}")
            placement_order.append((row, col))
            queen_board[row, col] = 1

        if capture_activations:
            return queen_board, placement_order, activations
        return queen_board, placement_order

    def solve_puzzle_diagnostic(self, puzzle: dict, ground_truth: np.ndarray = None,
                                 top_k: int = 10, verbose: bool = True) -> dict:
        """Solve puzzle with detailed diagnostic information for failure analysis."""
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]

        if ground_truth is None:
            ground_truth = self.solve_with_vanilla_backtracking(puzzle)

        correct_positions = set(zip(*np.where(ground_truth == 1)))

        queen_board = np.zeros((n, n), dtype=int)
        edge_index_dict = self._build_edge_index(region_board)
        placement_order = []
        step_diagnostics = []

        for step in range(n):
            row, col, _, logits_np = self.place_queen(
                region_board, queen_board, edge_index_dict,
                capture_activations=False,
                return_logits=True
            )

            remaining_correct = correct_positions - set(placement_order)

            flat_logits = logits_np.flatten()
            top_k_indices = np.argsort(flat_logits)[::-1][:top_k]
            top_k_predictions = []
            first_correct_rank = None

            for rank, idx in enumerate(top_k_indices, 1):
                pred_row, pred_col = idx // n, idx % n
                pred_logit = flat_logits[idx]
                is_correct = (pred_row, pred_col) in remaining_correct
                top_k_predictions.append({
                    'rank': rank,
                    'position': (pred_row, pred_col),
                    'logit': float(pred_logit),
                    'is_correct': is_correct
                })
                if is_correct and first_correct_rank is None:
                    first_correct_rank = rank

            step_info = {
                'step': step + 1,
                'queens_placed': list(placement_order),
                'remaining_correct': sorted(remaining_correct),
                'top_k_predictions': top_k_predictions,
                'first_correct_rank': first_correct_rank,
                'placement': (row, col)
            }
            step_diagnostics.append(step_info)

            if verbose:
                self._print_step_diagnostic(step_info)

            placement_order.append((row, col))
            queen_board[row, col] = 1

        is_correct = np.array_equal(queen_board, ground_truth)

        if verbose:
            print(f"\n{'='*50}")
            print(f"FINAL RESULT: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"{'='*50}")

        return {
            'solution': queen_board,
            'placement_order': placement_order,
            'correct': is_correct,
            'steps': step_diagnostics
        }

    def _print_step_diagnostic(self, step_info: dict) -> None:
        """Print formatted diagnostic output for a single step."""
        print(f"\n=== STEP {step_info['step']} ===")
        print(f"Queens placed so far: {step_info['queens_placed']}")
        print(f"Remaining correct positions: {step_info['remaining_correct']}")
        print(f"Top {len(step_info['top_k_predictions'])} predictions:")

        for pred in step_info['top_k_predictions']:
            pos = pred['position']
            logit = pred['logit']
            marker = '✓' if pred['is_correct'] else '✗'
            print(f"  Rank {pred['rank']:2d}: ({pos[0]},{pos[1]}) logit={logit:6.3f} {marker}")

        if step_info['first_correct_rank'] is not None:
            print(f"First correct position appears at rank: {step_info['first_correct_rank']}")
        else:
            print(f"First correct position appears at rank: >top-k")

        row, col = step_info['placement']
        print(f"Placing queen at model's top choice: ({row}, {col})")

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
        """Visualize the reasoning progression across queen placements with L and H-state activation heatmaps."""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        num_placements = len(activations)

        placed_queens = []

        for step_idx, act_dict in enumerate(activations):
            row, col = act_dict['placement']
            logit = act_dict['logit']

            has_H = 'H' in act_dict
            L_maps = act_dict['L']
            H_maps = act_dict.get('H', {})

            num_L_cols = len(L_maps)
            num_H_cols = len(H_maps) if has_H else 0
            num_cols = 1 + num_L_cols + num_H_cols

            fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
            if num_cols == 1:
                axes = [axes]

            fig.suptitle(f"Queen {step_idx + 1}/{num_placements}: Placement ({row}, {col}) | Logit: {logit:.3f}",
                        fontsize=14, fontweight='bold')

            ax = axes[0]
            if show_regions:
                self._draw_colored_board(ax, region_board, placed_queens, row, col)
            ax.set_title('Board State', fontsize=12, fontweight='bold')

            col_idx = 1
            for stage_name, heatmap in L_maps.items():
                ax = axes[col_idx]
                self._draw_heatmap(ax, heatmap, n, f"L-{stage_name.capitalize()}",
                                   placed_queens, row, col)
                col_idx += 1

            for stage_name, heatmap in H_maps.items():
                ax = axes[col_idx]
                self._draw_heatmap(ax, heatmap, n, f"H-{stage_name.capitalize()}",
                                   placed_queens, row, col, cmap='PuOr_r')
                col_idx += 1

            plt.tight_layout()

            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                filename = output_path / f"step_{step_idx:02d}_queen_{row}_{col}.png"
                plt.savefig(filename, dpi=100, bbox_inches='tight')
                logger.debug(f"Saved: {filename}")
                plt.close()
            else:
                plt.show()

            placed_queens.append((row, col))

        if output_dir:
            self._create_summary_image(output_dir, num_placements, n)

    def _draw_heatmap(self, ax, heatmap: np.ndarray, n: int, title: str,
                      placed_queens: List[Tuple[int, int]], current_row: int, current_col: int,
                      cmap: str = 'RdBu_r') -> None:
        """Draw a heatmap with queen markers."""
        im = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which='both', bottom=False, left=False,
                      labelbottom=False, labelleft=False)

        for prev_row, prev_col in placed_queens:
            ax.text(prev_col, prev_row, 'X', fontsize=24, ha='center', va='center',
                   color='black', fontweight='bold')

        ax.scatter([current_col], [current_row], marker='*', color='lime', s=1000,
                  edgecolors='white', linewidths=3, zorder=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _create_summary_image(self, output_dir, num_placements, n):
        """Concatenate all step visualizations vertically into a summary image."""
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
            logger.warning("No step images found for summary")
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
        logger.debug(f"Saved summary: {summary_path}")

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
        model_solution, _ = self.solve_puzzle(puzzle)
        backtrack_solution = self.solve_with_vanilla_backtracking(puzzle)
        return np.array_equal(model_solution, backtrack_solution)
