import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


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
            concat=True,            # default; dims multiply by heads
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
        self.linear  = nn.Linear(hidden_dim, 1)   # single logit per node

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        for conv in self.convs:
            x = self.relu(x + conv(x, edge_index))   # residual skip
            x = self.dropout(x)

        logits = self.linear(x).squeeze(-1)          # shape [N]
        return logits
