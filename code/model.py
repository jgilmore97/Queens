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
    input_dim   : int   – node-feature dimension (4 for your board encoding).
    hidden_dim  : int   – total hidden size after concatenating all heads.
    layer_count : int   – number of GAT layers (≥ 2).
    dropout     : float – dropout prob applied after each attention layer.
    heads       : int   – number of attention heads per layer.
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
    """
    Heterogeneous Graph Attention Network for Queens puzzle with BatchNorm.
    
    Uses different attention mechanisms for different constraint types:
    - line_constraint: row/column mutual exclusion
    - region_constraint: color region mutual exclusion  
    - diagonal_constraint: immediate diagonal adjacency
    """
    def __init__(self, input_dim, hidden_dim, layer_count, dropout, heads=2, use_batch_norm=True, input_injection_layers=None):
        super().__init__()
        
        # Each head outputs hidden_dim // heads channels; concatenated = hidden_dim
        head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = head_dim
        self.dropout_p = dropout
        self.use_batch_norm = use_batch_norm
        
        # Define edge types we expect
        self.edge_types = [
            ('cell', 'line_constraint', 'cell'),
            ('cell', 'region_constraint', 'cell'), 
            ('cell', 'diagonal_constraint', 'cell')
        ]

        # Store input injection configuration
        if input_injection_layers is None:
            # Default: inject input at later layers (e.g., last 2 layers)
            self.input_injection_layers = set(range(max(1, layer_count-2), layer_count))
        else:
            self.input_injection_layers = set(input_injection_layers)
        
        # First heterogeneous layer
        self.conv1 = HeteroConv({
            edge_type: pyg_nn.GATConv(
                in_channels=input_dim,
                out_channels=head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            for edge_type in self.edge_types
        }, aggr='sum')

        # Batch normalization for first layer
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(1, layer_count):
            hetero_conv = HeteroConv({
                edge_type: pyg_nn.GATConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(hetero_conv)
            
            # Add batch norm for each additional layer
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Include one graphformer layer to try to leverage global context
        self.graphformer = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], self.edge_types),
            heads=heads
        )
        if self.use_batch_norm:
            self.bn_graphformer = nn.BatchNorm1d(hidden_dim)
        
        self.linear = nn.Linear(hidden_dim, 1)

        # Add projection layers for input injection
        self.input_projections = nn.ModuleDict()
        for layer_idx in self.input_injection_layers:
            self.input_projections[str(layer_idx)] = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass for heterogeneous data with batch normalization.
        
        Args:
            x_dict: Dictionary with node features, e.g., {'cell': tensor}
            edge_index_dict: Dictionary with edge indices for each edge type
        """

        # Store original input features
        original_input = x_dict['cell']  # Save this!
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        if self.use_batch_norm:
            x_dict = {key: self.bn1(x) for key, x in x_dict.items()}
        
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        for i, conv in enumerate(self.convs):
            layer_idx = i + 1
            
            x_dict_new = conv(x_dict, edge_index_dict)
            
            if self.use_batch_norm:
                x_dict_new = {key: self.batch_norms[i](x) for key, x in x_dict_new.items()}
            
            # Input Injection: Add projected input features at specified layers
            if layer_idx in self.input_injection_layers:
                projected_input = self.input_projections[str(layer_idx)](original_input)
                x_dict_new['cell'] = x_dict_new['cell'] + projected_input
            
            # Standard residual connection between every layer
            x_dict = {
                key: self.relu(x_dict[key] + x_dict_new[key]) 
                for key in x_dict.keys()
            }
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # One graphformer layer to capture global context with residual connection
        x_dict_new = self.graphformer(x_dict, edge_index_dict)
        if self.use_batch_norm:
            x_dict_new = {key: self.bn_graphformer(x) for key, x in x_dict_new.items()}
        x_dict = {
            key: self.relu(x_dict[key] + x_dict_new[key]) 
            for key in x_dict.keys()
        }
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Final linear layer to get logits
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