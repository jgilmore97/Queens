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
    """Graph Attention Network using GATConv layers with multi-head attention."""
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
    """Heterogeneous GAT with constraint-specific attention and global context via HGT."""
    def __init__(self, input_dim, hidden_dim, layer_count, dropout, gat_heads=2, hgt_heads=6,
                 use_batch_norm=True, input_injection_layers=None):
        super().__init__()

        # Each GAT head outputs hidden_dim // gat_heads channels; concatenated = hidden_dim
        gat_head_dim = hidden_dim // gat_heads
        assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by gat_heads"

        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.hgt_heads = hgt_heads
        self.gat_head_dim = gat_head_dim
        self.dropout_p = dropout
        self.use_batch_norm = use_batch_norm
        self.layer_count = layer_count

        self.edge_types = [
            ('cell', 'line_constraint', 'cell'),
            ('cell', 'region_constraint', 'cell'),
            ('cell', 'diagonal_constraint', 'cell')
        ]

        if input_injection_layers is None:
            self.input_injection_layers = set(range(max(1, layer_count-2), layer_count))
        else:
            self.input_injection_layers = set(input_injection_layers)

        # Mid-sequence global context helps catch constraint violations early
        self.mid_global_layer = max(1, layer_count // 3)

        print(f" HeteroGAT configured:")
        print(f" GAT heads: {gat_heads} (constraint-specific attention)")
        print(f" HGT heads: {hgt_heads} (global context attention)")
        print(f" Mid-global context at layer {self.mid_global_layer}/{layer_count}")
        print(f" Expected to reduce Type 2 errors via enhanced global reasoning")

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

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)

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

        self.mid_graphformer = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], self.edge_types),
            heads=hgt_heads
        )
        if self.use_batch_norm:
            self.bn_mid_graphformer = nn.BatchNorm1d(hidden_dim)

        self.final_graphformer = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], self.edge_types),
            heads=hgt_heads
        )
        if self.use_batch_norm:
            self.bn_final_graphformer = nn.BatchNorm1d(hidden_dim)

        self.linear = nn.Linear(hidden_dim, 1)

        self.input_projections = nn.ModuleDict()
        for layer_idx in self.input_injection_layers:
            self.input_projections[str(layer_idx)] = nn.Linear(input_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        """Forward pass with multi-stage global context injection for Type 2 error prevention."""

        original_input = x_dict['cell']

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

            # Input injection: re-inject projected original features to preserve signal
            if layer_idx in self.input_injection_layers:
                projected_input = self.input_projections[str(layer_idx)](original_input)
                x_dict_new['cell'] = x_dict_new['cell'] + projected_input

            x_dict = {
                key: self.relu(x_dict[key] + x_dict_new[key])
                for key in x_dict.keys()
            }
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

            # Mid-sequence global context injection
            if layer_idx == self.mid_global_layer:
                x_dict_global = self.mid_graphformer(x_dict, edge_index_dict)
                if self.use_batch_norm:
                    x_dict_global = {key: self.bn_mid_graphformer(x) for key, x in x_dict_global.items()}

                x_dict = {
                    key: self.relu(x_dict[key] + x_dict_global[key])
                    for key in x_dict.keys()
                }
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Final global context refinement
        x_dict_final_global = self.final_graphformer(x_dict, edge_index_dict)
        if self.use_batch_norm:
            x_dict_final_global = {key: self.bn_final_graphformer(x) for key, x in x_dict_final_global.items()}
        x_dict = {
            key: self.relu(x_dict[key] + x_dict_final_global[key])
            for key in x_dict.keys()
        }
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

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


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.scale * x * (x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt())


class _InitialEmbed(nn.Module):
    """Projects input features to hidden_dim with normalization."""
    def __init__(self, in_dim: int, hidden_dim: int, p_drop: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = _RMSNorm(hidden_dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x_cell):
        h = torch.nn.functional.gelu(self.proj(x_cell))
        return self.drop(self.norm(h))


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions node features on global z_H."""
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
    """L-module: weight-tied micro-step with GAT->GAT->HGT and FiLM conditioning."""
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

        self.hgt = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], edge_types),
            heads=hgt_heads
        )
        self.bn_hgt = nn.BatchNorm1d(hidden_dim) if self.use_bn else None

        if self.use_input_injection:
            self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        self.film = _FiLM(hidden_dim)

    def _res_block(self, x_old, x_new, bn):
        if bn is not None:
            x_new = bn(x_new)
        return self.relu(x_old + self.drop(x_new))

    def forward(self, x_cell, edge_index_dict, z_h, x_cell_embedded=None):
        """
        x_cell: [C, d] current node states
        z_h: [d] global vector (fixed during this micro-step)
        x_cell_embedded: [C, d] embedded original input for injection
        """
        x_dict = {'cell': x_cell}

        x1 = self.hconv1(x_dict, edge_index_dict)['cell']
        x1 = self._res_block(x_cell, x1, self.bn1)

        x2 = self.hconv2({'cell': x1}, edge_index_dict)['cell']
        x2 = self._res_block(x1, x2, self.bn2)

        x3 = self.hgt({'cell': x2}, edge_index_dict)['cell']
        if self.bn_hgt is not None:
            x3 = self.bn_hgt(x3)
        x3 = self._res_block(x2, x3, None)

        if self.use_input_injection and x_cell_embedded is not None:
            x3 = x3 + self.input_proj(x_cell_embedded)

        x3 = self.film(x3, z_h)
        return x3


class _HModule(nn.Module):
    """H-module: updates global state z_H via multi-head attention pooling over nodes."""
    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)  # z_prev -> queries
        self.key = nn.Linear(hidden_dim, hidden_dim)    # nodes -> keys
        self.value = nn.Linear(hidden_dim, hidden_dim)  # nodes -> values

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = _RMSNorm(hidden_dim)

    def forward(self, nodes, z_prev, return_attention=False):
        """
        nodes: [C, d] all node states
        z_prev: [d] previous global state
        """
        C, d = nodes.shape

        Q = self.query(z_prev).view(1, self.num_heads, self.head_dim)  # [1, H, d/H]
        K = self.key(nodes).view(C, self.num_heads, self.head_dim)     # [C, H, d/H]
        V = self.value(nodes).view(C, self.num_heads, self.head_dim)   # [C, H, d/H]

        attn_scores = torch.einsum('qhd,chd->hc', Q, K) / (self.head_dim ** 0.5)  # [H, C]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [H, C]

        pooled_heads = torch.einsum('hc,chd->hd', attn_weights, V)  # [H, d/H]
        pooled = pooled_heads.reshape(self.hidden_dim)  # [d]

        z = torch.cat([pooled, z_prev], dim=-1)  # [2d]
        z_out = self.norm(self.mlp(z))  # [d]

        if return_attention:
            avg_attention = attn_weights.mean(dim=0)  # [C]
            return z_out, avg_attention
        return z_out


class _Readout(nn.Module):
    """Produces per-cell logits conditioned on global z_H."""
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
    Hierarchical Reasoning Model with L-module (fast local) and H-module (slow global).
    Based on Wang et al., 2025.
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

        if z_init == "learned":
            self.z0 = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        else:
            self.register_buffer("z0", torch.zeros(hidden_dim), persistent=False)

    def forward(self, x_dict, edge_index_dict, return_intermediates=False):
        """
        Forward pass through hierarchical L/H architecture.
        Returns logits [C] or (logits, intermediates) if return_intermediates=True.
        """
        x_in = x_dict['cell']
        nodes_embedded = self.embed(x_in)
        nodes = nodes_embedded
        z = self.z0

        if return_intermediates:
            L_states = []
            H_attention = []
            z_H_history = [z.detach().cpu()]
            film_params = []

        for cycle_idx in range(self.n_cycles):
            for micro_idx in range(self.t_micro):
                if return_intermediates:
                    gamma, beta = self.l_block.film.mlp(z).chunk(2, dim=-1)
                    film_params.append({
                        'cycle': cycle_idx,
                        'micro': micro_idx,
                        'gamma': gamma.detach().cpu(),
                        'beta': beta.detach().cpu()
                    })

                nodes = self.l_block(
                    nodes,
                    edge_index_dict,
                    z,
                    x_cell_embedded=nodes_embedded if self.use_input_injection else None
                )

                if return_intermediates:
                    L_states.append(nodes.detach().cpu())

            if return_intermediates:
                z, attention = self.h_mod(nodes, z, return_attention=True)
                H_attention.append(attention.detach().cpu())
                z_H_history.append(z.detach().cpu())
            else:
                z = self.h_mod(nodes, z, return_attention=False)

        logits = self.readout(nodes, z)

        if return_intermediates:
            intermediates = {
                'L_states': L_states,           # List of tensors (t_micro * n_cycles)
                'H_attention': H_attention,     # List of tensors (n_cycles)
                'z_H_history': z_H_history,     # List of tensors (n_cycles + 1)
                'film_params': film_params,     # List of dicts with gamma, beta
                'final_logits': logits.detach().cpu(),
                'board_size': int(x_in.shape[0] ** 0.5)
            }
            return logits, intermediates

        return logits
