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
    input_dim   : int   — node-feature dimension (4 for your board encoding).
    hidden_dim  : int   — total hidden size after concatenating all heads.
    layer_count : int   — number of GAT layers (≥ 2).
    dropout     : float — dropout prob applied after each attention layer.
    heads       : int   — number of attention heads per layer.
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
        
        print(f"🧠 HeteroGAT configured:")
        print(f"   GAT heads: {gat_heads} (constraint-specific attention)")
        print(f"   HGT heads: {hgt_heads} (global context attention)")
        print(f"   Mid-global context at layer {self.mid_global_layer}/{layer_count}")
        print(f"   Expected to reduce Type 2 errors via enhanced global reasoning")
        
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