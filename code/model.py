import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HeteroConv, HGTConv


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, dropout):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim, add_self_loops=True)
        self.convs = nn.ModuleList(
            pyg_nn.GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
            for _ in range(1, layer_count)
        )
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        for conv in self.convs:
            x = self.relu(x + conv(x, edge_index))
            x = self.dropout(x)
        logits = self.linear(x).squeeze(-1)
        return logits  

class GAT(nn.Module):
    """
    Graph-Attention version (GATConv instead of GCNConv).

    Parameters
    ----------
    input_dim   : int   - node-feature dimension (4 for your board encoding).
    hidden_dim  : int   - total hidden size after concatenating all heads.
    layer_count : int   - number of GAT layers (>= 2).
    dropout     : float - dropout prob applied after each attention layer.
    heads       : int   - number of attention heads per layer.
    """
    def __init__(self, input_dim, hidden_dim, layer_count, dropout, heads=2):
        super().__init__()

        # Each head outputs hidden_dim // heads channels; concatenated = hidden_dim
        head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.conv1 = pyg_nn.GATConv(
            in_channels=input_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )

        self.convs = nn.ModuleList(
            pyg_nn.GATConv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(1, layer_count)
        )

        self.relu    = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        for conv in self.convs:
            x = self.relu(x + conv(x, edge_index))
            x = self.dropout(x)

        logits = self.linear(x).squeeze(-1)
        return logits

class HeteroGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, dropout, gat_heads=2, hgt_heads=6, 
                 use_batch_norm=True, input_injection_layers=None):
        super().__init__()
        
        # Each GAT head outputs hidden_dim // gat_heads channels
        gat_head_dim = hidden_dim // gat_heads
        assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by gat_heads"
        
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.hgt_heads = hgt_heads
        self.gat_head_dim = gat_head_dim
        self.dropout_p = dropout
        self.use_batch_norm = use_batch_norm
        self.layer_count = layer_count
        
        # Define edge types we expect
        self.edge_types = [
            ('cell', 'line_constraint', 'cell'),
            ('cell', 'region_constraint', 'cell'), 
            ('cell', 'diagonal_constraint', 'cell')
        ]

        # Store input injection configuration
        if input_injection_layers is None:
            self.input_injection_layers = set(range(max(1, layer_count-2), layer_count))
        else:
            self.input_injection_layers = set(input_injection_layers)
        
        # Configure mid-sequence global context layer
        self.mid_global_layer = max(1, layer_count // 3)
        
        print(f" HeteroGAT configured:")
        print(f" GAT heads: {gat_heads} (constraint-specific attention)")
        print(f" HGT heads: {hgt_heads} (global context attention)")
        print(f" Mid-global context at layer {self.mid_global_layer}/{layer_count}")
        print(f" Expected to reduce Type 2 errors via enhanced global reasoning")
        
        # First heterogeneous layer 
        self.conv1 = HeteroConv({
            edge_type: pyg_nn.GATConv(
                in_channels=input_dim,
                out_channels=gat_head_dim,
                heads=gat_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for edge_type in self.edge_types
        }, aggr='sum')

        # Batch normalization for first layer
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Additional GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(1, layer_count):
            hetero_conv = HeteroConv({
                edge_type: pyg_nn.GATConv(
                    in_channels=hidden_dim,
                    out_channels=gat_head_dim,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(hetero_conv)
            
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # MID-SEQUENCE GLOBAL CONTEXT LAYER
        self.mid_graphformer = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], self.edge_types),
            heads=hgt_heads 
        )
        if self.use_batch_norm:
            self.bn_mid_graphformer = nn.BatchNorm1d(hidden_dim)

        # FINAL GLOBAL CONTEXT LAYER
        self.final_graphformer = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], self.edge_types),
            heads=hgt_heads
        )
        if self.use_batch_norm:
            self.bn_final_graphformer = nn.BatchNorm1d(hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, 1)

        # Add projection layers for input injection
        self.input_projections = nn.ModuleDict()
        for layer_idx in self.input_injection_layers:
            self.input_projections[str(layer_idx)] = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with multi-stage global context injection.
        
        Key Enhancement: Mid-sequence global context to catch Type 2 errors early
        
        Args:
            x_dict: Dictionary with node features, e.g., {'cell': tensor}
            edge_index_dict: Dictionary with edge indices for each edge type
        """

        # Store original input features for injection
        original_input = x_dict['cell']
        
        # STAGE 1: Initial heterogeneous attention
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        if self.use_batch_norm:
            x_dict = {key: self.bn1(x) for key, x in x_dict.items()}
        
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # STAGE 2: Progressive GAT layers with mid-sequence global injection
        for i, conv in enumerate(self.convs):
            layer_idx = i + 1
            
            # Regular heterogeneous attention step
            x_dict_new = conv(x_dict, edge_index_dict)
            
            if self.use_batch_norm:
                x_dict_new = {key: self.batch_norms[i](x) for key, x in x_dict_new.items()}
            
            # Input Injection: Add projected input features at specified layers
            if layer_idx in self.input_injection_layers:
                projected_input = self.input_projections[str(layer_idx)](original_input)
                x_dict_new['cell'] = x_dict_new['cell'] + projected_input
            
            # Standard residual connection
            x_dict = {
                key: self.relu(x_dict[key] + x_dict_new[key]) 
                for key in x_dict.keys()
            }
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
            
            # MID-SEQUENCE GLOBAL CONTEXT INJECTION - Type 2 Error Prevention
            if layer_idx == self.mid_global_layer:
                # Apply global context via HGT transformer
                x_dict_global = self.mid_graphformer(x_dict, edge_index_dict)
                if self.use_batch_norm:
                    x_dict_global = {key: self.bn_mid_graphformer(x) for key, x in x_dict_global.items()}
                
                # Integrate global context with residual connection
                x_dict = {
                    key: self.relu(x_dict[key] + x_dict_global[key]) 
                    for key in x_dict.keys()
                }
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # STAGE 3: Final global context refinement
        x_dict_final_global = self.final_graphformer(x_dict, edge_index_dict)
        if self.use_batch_norm:
            x_dict_final_global = {key: self.bn_final_graphformer(x) for key, x in x_dict_final_global.items()}
        x_dict = {
            key: self.relu(x_dict[key] + x_dict_final_global[key]) 
            for key in x_dict.keys()
        }
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # STAGE 4: Final prediction
        cell_features = x_dict['cell']
        logits = self.linear(cell_features).squeeze(-1)
        
        return logits

    def get_attention_weights(self, x_dict, edge_index_dict, layer_idx=0):
        if layer_idx == 0:
            conv_layer = self.conv1
        elif layer_idx <= len(self.convs):
            conv_layer = self.convs[layer_idx - 1]
        else:
            raise ValueError(f"Layer {layer_idx} doesn't exist")
        
        attention_weights = {}
        
        print(f"Attention extraction for layer {layer_idx} would go here")
        print("Edge types:", list(edge_index_dict.keys()))
        
        return attention_weights

# ======================================================================
# HRM-style hierarchical reasoning model
# Inspired by "Hierarchical Reasoning Model" (Wang et al., 2025)
# Two-level architecture: L-module (fast local) + H-module (slow global)
# ======================================================================

class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    
    def forward(self, x):
        return self.scale * x * (x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt())


class _InitialEmbed(nn.Module):
    """Initial embedding layer: projects input features to hidden_dim"""
    def __init__(self, in_dim: int, hidden_dim: int, p_drop: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = _RMSNorm(hidden_dim)
        self.drop = nn.Dropout(p_drop)
    
    def forward(self, x_cell):
        h = torch.nn.functional.gelu(self.proj(x_cell))
        return self.drop(self.norm(h))


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions node features on global z_H"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        )
    
    def forward(self, nodes, z_h):
        gamma, beta = self.mlp(z_h).chunk(2, dim=-1)
        return gamma * nodes + beta


class _LBlock(nn.Module):
    """
    L-module: One weight-tied micro-step
    Architecture: GAT -> GAT -> HGT with FiLM conditioning from z_H
    """
    def __init__(
        self,
        hidden_dim: int,
        gat_heads: int,
        hgt_heads: int,
        dropout: float,
        use_batch_norm: bool,
        edge_types,
        use_input_injection: bool,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2)
        self.use_bn = use_batch_norm
        self.use_input_injection = use_input_injection

        gat_head_dim = hidden_dim // max(1, gat_heads)
        assert hidden_dim % max(1, gat_heads) == 0, "hidden_dim must be divisible by gat_heads"

        # Two heterogeneous GAT layers
        self.hconv1 = HeteroConv({
            et: pyg_nn.GATConv(
                in_channels=hidden_dim,
                out_channels=gat_head_dim,
                heads=gat_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            ) for et in edge_types
        }, aggr='sum')

        self.hconv2 = HeteroConv({
            et: pyg_nn.GATConv(
                in_channels=hidden_dim,
                out_channels=gat_head_dim,
                heads=gat_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            ) for et in edge_types
        }, aggr='sum')

        self.bn1 = nn.BatchNorm1d(hidden_dim) if self.use_bn else None
        self.bn2 = nn.BatchNorm1d(hidden_dim) if self.use_bn else None

        # Heterogeneous global context layer
        self.hgt = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], edge_types),
            heads=hgt_heads
        )
        self.bn_hgt = nn.BatchNorm1d(hidden_dim) if self.use_bn else None

        # Input injection (embedded input -> hidden)
        if self.use_input_injection:
            self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # FiLM conditioning from z_H
        self.film = _FiLM(hidden_dim)

    def _res_block(self, x_old, x_new, bn):
        if bn is not None:
            x_new = bn(x_new)
        return self.relu(x_old + self.drop(x_new))

    def forward(self, x_cell, edge_index_dict, z_h, x_cell_embedded=None):
        """
        x_cell: [C, d] current node states
        edge_index_dict: hetero edges
        z_h: [d] global vector (fixed during this micro-step)
        x_cell_embedded: [C, d] embedded original input for injection
        """
        x_dict = {'cell': x_cell}

        # GAT layer 1
        x1 = self.hconv1(x_dict, edge_index_dict)['cell']
        x1 = self._res_block(x_cell, x1, self.bn1)

        # GAT layer 2
        x2 = self.hconv2({'cell': x1}, edge_index_dict)['cell']
        x2 = self._res_block(x1, x2, self.bn2)

        # HGT global context
        x3 = self.hgt({'cell': x2}, edge_index_dict)['cell']
        if self.bn_hgt is not None:
            x3 = self.bn_hgt(x3)
        x3 = self._res_block(x2, x3, None)

        # Input injection (embedded input)
        if self.use_input_injection and x_cell_embedded is not None:
            x3 = x3 + self.input_proj(x_cell_embedded)

        # FiLM conditioning with z_H
        x3 = self.film(x3, z_h)
        return x3


class _HModule(nn.Module):
    """H-module with multi-head attention pooling"""
    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention for pooling
        self.query = nn.Linear(hidden_dim, hidden_dim)  # z_prev -> queries
        self.key = nn.Linear(hidden_dim, hidden_dim)    # nodes -> keys
        self.value = nn.Linear(hidden_dim, hidden_dim)  # nodes -> values
        
        # MLP to update z_H
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = _RMSNorm(hidden_dim)
    
    def forward(self, nodes, z_prev, return_attention=False):
        """
        nodes: [C, d] - all node states
        z_prev: [d] - previous global state
        return_attention: if True, also return averaged attention weights
        """
        C, d = nodes.shape
        
        # Generate Q, K, V
        Q = self.query(z_prev).view(1, self.num_heads, self.head_dim)  # [1, H, d/H]
        K = self.key(nodes).view(C, self.num_heads, self.head_dim)     # [C, H, d/H]
        V = self.value(nodes).view(C, self.num_heads, self.head_dim)   # [C, H, d/H]
        
        # Compute attention scores per head
        attn_scores = torch.einsum('qhd,chd->hc', Q, K) / (self.head_dim ** 0.5)  # [H, C]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [H, C]
        
        # Weighted sum of values per head
        pooled_heads = torch.einsum('hc,chd->hd', attn_weights, V)  # [H, d/H]
        pooled = pooled_heads.reshape(self.hidden_dim)  # [d]
        
        # Update z_H
        z = torch.cat([pooled, z_prev], dim=-1)  # [2d]
        z_out = self.norm(self.mlp(z))  # [d]
        
        if return_attention:
            # Average attention weights across heads
            avg_attention = attn_weights.mean(dim=0)  # [C]
            return z_out, avg_attention
        return z_out


class _Readout(nn.Module):
    """Readout: per-cell logits conditioned on global z_H"""
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, nodes, z_h):
        z_tiled = z_h.unsqueeze(0).expand_as(nodes)
        return self.mlp(torch.cat([nodes, z_tiled], dim=-1)).squeeze(-1)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model for Queens puzzle
    
    Architecture:
    - L-module: Weight-tied recurrent block (GAT->GAT->HGT)
    - H-module: Global state manager with multi-head attention pooling
    - Hierarchical convergence: L converges locally, H updates context
    
    Based on "Hierarchical Reasoning Model" (Wang et al., 2025)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gat_heads: int = 2,
        hgt_heads: int = 6,
        dropout: float = 0.10,
        use_batch_norm: bool = True,
        n_cycles: int = 2,
        t_micro: int = 2,
        use_input_injection: bool = True,
        z_init: str = "zeros",
        h_pooling_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.t_micro = t_micro
        self.use_input_injection = use_input_injection

        self.edge_types = [
            ('cell', 'line_constraint', 'cell'),
            ('cell', 'region_constraint', 'cell'),
            ('cell', 'diagonal_constraint', 'cell')
        ]

        print(f"HRM configured:")
        print(f"Cycles (H-updates): {n_cycles}")
        print(f"Micro-steps per cycle: {t_micro}")
        print(f"Total L-steps: {n_cycles * t_micro}")
        print(f"GAT heads: {gat_heads}, HGT heads: {hgt_heads}")
        print(f"H-pooling heads: {h_pooling_heads}")
        print(f"Input injection: {use_input_injection}")

        # Modules
        self.embed = _InitialEmbed(input_dim, hidden_dim, dropout)
        self.l_block = _LBlock(
            hidden_dim=hidden_dim,
            gat_heads=gat_heads,
            hgt_heads=hgt_heads,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            edge_types=self.edge_types,
            use_input_injection=use_input_injection,
        )
        self.h_mod = _HModule(hidden_dim, dropout, num_heads=h_pooling_heads)
        self.readout = _Readout(hidden_dim, dropout)

        # z_H initialization
        if z_init == "learned":
            self.z0 = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        else:
            self.register_buffer("z0", torch.zeros(hidden_dim), persistent=False)

    def forward(self, x_dict, edge_index_dict, return_intermediates=False):
        """
        x_dict: {'cell': [C, input_dim]}
        edge_index_dict: hetero edges
        return_intermediates: if True, return dict with intermediate states
        Returns: logits [C] or (logits, intermediates) if return_intermediates=True
        """
        x_in = x_dict['cell']
        nodes_embedded = self.embed(x_in)
        nodes = nodes_embedded
        z = self.z0
        
        # Storage for intermediates
        if return_intermediates:
            L_states = []
            H_attention = []
        
        for cycle_idx in range(self.n_cycles):
            # L micro-steps (weight-tied)
            for micro_idx in range(self.t_micro):
                nodes = self.l_block(
                    nodes, 
                    edge_index_dict, 
                    z, 
                    x_cell_embedded=nodes_embedded if self.use_input_injection else None
                )
                
                if return_intermediates:
                    # Store L-module state after each micro-step
                    L_states.append(nodes.detach().cpu())

            # H update with attention pooling
            if return_intermediates:
                z, attention = self.h_mod(nodes, z, return_attention=True)
                H_attention.append(attention.detach().cpu())
            else:
                z = self.h_mod(nodes, z, return_attention=False)

        # Final readout conditioned on z_H
        logits = self.readout(nodes, z)
        
        if return_intermediates:
            intermediates = {
                'L_states': L_states,  # List of 6 tensors (2 micro × 3 cycles)
                'H_attention': H_attention,  # List of 3 tensors
                'final_logits': logits.detach().cpu(),
                'board_size': int(x_in.shape[0] ** 0.5)  # Assuming square board
            }
            return logits, intermediates
        
        return logits