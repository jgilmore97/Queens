import torch
import numpy as np
from typing import Dict

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
    
    def _build_edge_index(self, region_board: np.ndarray) -> Dict[str, torch.Tensor]:
        edge_index_dict = build_heterogeneous_edge_index(region_board)
        
        for edge_type, edge_index in edge_index_dict.items():
            edge_index_dict[edge_type] = edge_index.to(self.device)
        
        return edge_index_dict

    def place_queen(self, region_board: np.ndarray, partial_board: np.ndarray, 
                    edge_index_dict: Dict[str, torch.Tensor]):
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
            logits = self.model(x_dict, edge_index_dict_formatted)

        logits_np = logits.cpu().numpy().reshape(n, n)
        flat_logits = logits_np.flatten()
        top_idx = np.argmax(flat_logits)
        top_logit = flat_logits[top_idx]
        top_row, top_col = top_idx // n, top_idx % n
        return top_row, top_col, top_logit

    def solve_puzzle(self, puzzle: dict):
        region_board = np.array(puzzle['region'])
        n = region_board.shape[0]
        queen_board = np.zeros((n, n), dtype=int)
        edge_index_dict = self._build_edge_index(region_board)

        for step in range(n):
            row, col, top_logit = self.place_queen(region_board, queen_board, edge_index_dict)
            print(f"Placing queen at: ({row}, {col}) with logit score: {top_logit:.3f}")
            queen_board[row, col] = 1
        
        return queen_board
    
    @staticmethod
    def solve_with_vanilla_backtracking(puzzle: dict):
        region_board = np.array(puzzle['region'])
        positions, board = solve_queens(region_board)
        return board
    
    def evaluate_solver(self, puzzle: dict) -> bool:
        model_solution = self.solve_puzzle(puzzle)
        backtrack_solution = self.solve_with_vanilla_backtracking(puzzle)
        return np.array_equal(model_solution, backtrack_solution)