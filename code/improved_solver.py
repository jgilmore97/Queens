import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from model import HRM, HeteroGAT
from data_loader import build_heterogeneous_edge_index
from board_manipulation import solve_queens


class Solver:
    def __init__(self, model_path, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path).to(device)
        self.max_regions = 11
        self.model.eval()

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['config_dict']
        
        is_hrm = model_config.get('model_type') == 'HRM' or 'n_cycles' in model_config
        
        if is_hrm:
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
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Model config: {model_config}")

        return model

    def _build_node_features(self, region_board: np.ndarray, queen_board: np.ndarray) -> torch.Tensor:
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
        """
        Convert HRM intermediates (L-states, H-attention) to per-cycle activation heatmaps.
        
        Parameters
        ----------
        intermediates : dict
            Dict with 'L_states' and 'H_attention' from HRM forward pass
            L_states: list of [C, d] tensors (6 total, 2 micro × 3 cycles)
            H_attention: list of [C] tensors (3 total, one per cycle)
        n : int
            Board dimension (n×n)
        activation_metric : str
            Metric for computing per-cell activation: 'l2_norm', 'mean_embedding', or 'max_embedding'
        
        Returns
        -------
        dict with keys:
            'L': {'early', 'mid', 'late', 'full'} - [n, n] activation heatmaps
            'H': {'early', 'mid', 'late', 'full'} - [n, n] attention heatmaps
        """
        L_states = intermediates['L_states']  # List of 6 tensors [C, d]
        H_attention = intermediates['H_attention']  # List of 3 tensors [C]
        
        # Process L-states: group by cycle, compute activation per cell
        L_early = self._compute_cycle_activation(L_states[0:2], n, activation_metric)      # Micro 0-1 of cycle 0
        L_mid = self._compute_cycle_activation(L_states[2:4], n, activation_metric)        # Micro 0-1 of cycle 1
        L_late = self._compute_cycle_activation(L_states[4:6], n, activation_metric)       # Micro 0-1 of cycle 2
        L_full = self._compute_cycle_activation(L_states, n, activation_metric)            # All 6 states
        
        # Process H-attention: reshape and normalize
        H_early = H_attention[0].cpu().numpy().reshape(n, n)           # After cycle 0
        H_mid = H_attention[1].cpu().numpy().reshape(n, n)             # After cycle 1
        H_late = H_attention[2].cpu().numpy().reshape(n, n)            # After cycle 2
        H_full = np.mean([h.cpu().numpy() for h in H_attention], axis=0).reshape(n, n)
        
        return {
            'L': {
                'early': L_early,
                'mid': L_mid,
                'late': L_late,
                'full': L_full
            },
            'H': {
                'early': H_early,
                'mid': H_mid,
                'late': H_late,
                'full': H_full
            }
        }
    
    def _compute_cycle_activation(self, state_list: List[torch.Tensor], n: int, 
                                 activation_metric: str = 'l2_norm') -> np.ndarray:
        """
        Compute per-cell activation from L-states, averaged across timesteps.
        
        Parameters
        ----------
        state_list : List[torch.Tensor]
            List of [C, d] embeddings (C = n², d = hidden_dim)
        n : int
            Board dimension
        activation_metric : str
            Metric for aggregation: 'l2_norm', 'mean_embedding', or 'max_embedding'
        
        Returns
        -------
        np.ndarray [n, n] - per-cell activation heatmap (normalized to [0, 1] for L2/max, diverging for mean)
        """
        # Stack and average across timesteps
        stacked = torch.stack(state_list, dim=0)  # [T, C, d]
        mean_state = stacked.mean(dim=0)  # [C, d]
        
        # Compute activation per cell based on metric
        if activation_metric == 'l2_norm':
            # L2-norm: magnitude of embedding vector
            activations = torch.norm(mean_state, dim=1)  # [C]
        elif activation_metric == 'mean_embedding':
            # Mean across embedding dimensions: captures directional bias
            activations = mean_state.mean(dim=1)  # [C]
        elif activation_metric == 'max_embedding':
            # Max activation across dimensions: strongest signal per cell
            activations = torch.max(torch.abs(mean_state), dim=1)[0]  # [C]
        else:
            raise ValueError(f"Unknown activation_metric: {activation_metric}")
        
        # Reshape to board
        heatmap = activations.cpu().numpy().reshape(n, n)
        
        # Normalize based on metric type
        if activation_metric == 'mean_embedding':
            # For mean_embedding, use centered normalization (diverging around 0)
            heatmap_abs_max = np.max(np.abs(heatmap))
            if heatmap_abs_max > 1e-6:
                heatmap = heatmap / heatmap_abs_max
            # Shift to [0, 1] for visualization (0.5 = neutral)
            heatmap = (heatmap + 1.0) / 2.0
        else:
            # For L2 and max, normalize to [0, 1]
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max > heatmap_min:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        return heatmap

    def place_queen(self, region_board: np.ndarray, partial_board: np.ndarray, 
                    edge_index_dict: Dict[str, torch.Tensor], 
                    capture_activations: bool = False,
                    activation_metric: str = 'l2_norm') -> Tuple:
        """
        Place a queen on the board using model predictions.
        
        Parameters
        ----------
        region_board : np.ndarray
            Color region map [n, n]
        partial_board : np.ndarray
            Current queen placements [n, n]
        edge_index_dict : Dict[str, torch.Tensor]
            Heterogeneous graph edges
        capture_activations : bool
            If True, capture and return intermediate activations from HRM
        activation_metric : str
            Metric for computing activations: 'l2_norm', 'mean_embedding', or 'max_embedding'
        
        Returns
        -------
        If capture_activations=False:
            (row, col, logit)
        If capture_activations=True:
            (row, col, logit, activation_dict) where activation_dict contains
            per-cycle L-state and H-attention heatmaps
        """
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

        logits_np = logits.cpu().numpy().reshape(n, n)
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
        """
        Solve a Queens puzzle autoregressively.
        
        Parameters
        ----------
        puzzle : dict
            Puzzle with 'region' key containing [n, n] region IDs
        capture_activations : bool
            If True, capture layer activations for each queen placement
        activation_metric : str
            Metric for computing activations: 'l2_norm', 'mean_embedding', or 'max_embedding'
        
        Returns
        -------
        If capture_activations=False:
            queen_board [n, n]
        If capture_activations=True:
            (queen_board, activation_list) where activation_list contains
            one activation_dict per placement
        """
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        queen_board = np.zeros((n, n), dtype=int)
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
        edge_index_dict = build_heterogeneous_edge_index(region_board)
        
        for edge_type, edge_index in edge_index_dict.items():
            edge_index_dict[edge_type] = edge_index.to(self.device)
        
        return edge_index_dict
    
    def visualize_solution(self, puzzle: dict, solution: np.ndarray, 
                          activations: List[dict], output_dir: Optional[str] = None,
                          show_regions: bool = False, activation_metric: str = 'l2_norm') -> None:
        """
        Visualize the reasoning progression across queen placements.
        
        Parameters
        ----------
        puzzle : dict
            Puzzle dict with 'region' key
        solution : np.ndarray
            Solved board [n, n] with queen positions
        activations : List[dict]
            Activation data for each placement from solve_puzzle()
        output_dir : Optional[str]
            Directory to save visualizations. If None, displays interactively.
        show_regions : bool
            If True, overlay regional color coding at low opacity
        activation_metric : str
            Activation metric used ('l2_norm', 'mean_embedding', 'max_embedding')
            Used to select appropriate colormap
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        num_placements = len(activations)
        
        # Select colormap based on metric
        if activation_metric == 'mean_embedding':
            L_cmap = 'RdBu_r'  # Diverging colormap for signed values
            H_cmap = 'RdBu_r'
        else:
            L_cmap = 'hot'  # Sequential colormap for magnitude
            H_cmap = 'cool'
        
        # Create region overlay if requested
        region_overlay = None
        if show_regions:
            region_overlay = self._create_region_overlay(region_board, n)
        
        for step_idx, act_dict in enumerate(activations):
            row, col = act_dict['placement']
            logit = act_dict['logit']
            
            # Create figure with subplots: 2 rows (L and H), 4 cols (early/mid/late/full)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"Queen {step_idx + 1}/{num_placements}: Placement ({row}, {col}) | Logit: {logit:.3f}", 
                        fontsize=14, fontweight='bold')
            
            # Row 0: L-module activations
            L_maps = act_dict['L']
            for col_idx, (stage_name, heatmap) in enumerate([
                ('Early', L_maps['early']),
                ('Mid', L_maps['mid']),
                ('Late', L_maps['late']),
                ('Full', L_maps['full'])
            ]):
                ax = axes[0, col_idx]
                im = ax.imshow(heatmap, cmap=L_cmap, vmin=0, vmax=1)
                
                # Overlay regions if requested
                if show_regions:
                    ax.imshow(region_overlay, alpha=0.15, cmap='tab20')
                
                ax.set_title(f"L-{stage_name}", fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Mark placement with cyan star
                ax.scatter([col], [row], marker='*', color='cyan', s=500, edgecolors='white', linewidths=2)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Row 1: H-module attention
            H_maps = act_dict['H']
            for col_idx, (stage_name, heatmap) in enumerate([
                ('Early', H_maps['early']),
                ('Mid', H_maps['mid']),
                ('Late', H_maps['late']),
                ('Full', H_maps['full'])
            ]):
                ax = axes[1, col_idx]
                im = ax.imshow(heatmap, cmap=H_cmap, vmin=0, vmax=1)
                
                # Overlay regions if requested
                if show_regions:
                    ax.imshow(region_overlay, alpha=0.15, cmap='tab20')
                
                ax.set_title(f"H-{stage_name}", fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Mark placement with yellow star
                ax.scatter([col], [row], marker='*', color='yellow', s=500, edgecolors='white', linewidths=2)
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
    
    def _create_region_overlay(self, region_board: np.ndarray, n: int) -> np.ndarray:
        """
        Create a region overlay for visualization.
        
        Parameters
        ----------
        region_board : np.ndarray
            [n, n] region IDs
        n : int
            Board dimension
        
        Returns
        -------
        np.ndarray [n, n] - normalized region IDs for colormapping
        """
        # Normalize region IDs to [0, 1] for visualization
        overlay = region_board.astype(np.float32)
        overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
        return overlay
    
    def evaluate_solver(self, puzzle: dict) -> bool:
        model_solution = self.solve_puzzle(puzzle)
        backtrack_solution = self.solve_with_vanilla_backtracking(puzzle)
        return np.array_equal(model_solution, backtrack_solution)