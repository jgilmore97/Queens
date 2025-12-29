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
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, nodes, z_h, batch_size: int = None):
        """
        Args:
            nodes: [N, d] node features
            z_h: [d] shared vector OR [B, d] per-graph vectors
            batch_size: Required only when using fixed batch mode with per-graph z
        """
        d = self.hidden_dim

        if z_h.dim() == 1:
            # Shared z: broadcast directly to all nodes
            gb = self.fc2(self.act(self.fc1(z_h)))  # [2d]
            gamma, beta = gb.chunk(2, dim=-1)  # each [d]
            gamma = 0.1 * torch.tanh(gamma)
            beta = 0.1 * torch.tanh(beta)
            return (1.0 + gamma) * nodes + beta  # broadcasts [d] over [N, d]

        # Per-graph z: requires fixed batch size
        B = batch_size
        N = nodes.shape[0]
        assert B is not None, "batch_size required for per-graph z"
        assert N % B == 0, "Fixed batch mode requires uniform nodes per graph"
        C = N // B

        gb = self.fc2(self.act(self.fc1(z_h)))  # [B, 2d]
        gamma, beta = gb.chunk(2, dim=-1)  # each [B, d]
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        nodes = nodes.view(B, C, d)
        out = (1.0 + gamma)[:, None, :] * nodes + beta[:, None, :]
        return out.view(N, d)


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

        self.norm1 = _RMSNorm(hidden_dim)
        self.norm2 = _RMSNorm(hidden_dim)

        self.hgt = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['cell'], edge_types),
            heads=hgt_heads
        )
        self.norm_hgt = _RMSNorm(hidden_dim)

        if self.use_input_injection:
            self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        self.film = _FiLM(hidden_dim)

    def _res_block(self, x_old, x_new, norm):
        x_new = norm(x_new)
        return self.relu(x_old + self.drop(x_new))

    def forward(self, x_cell, edge_index_dict, z_h, batch_size: int = None, x_cell_embedded=None):
        """
        Args:
            x_cell: [N, d] current node states
            z_h: [d] shared vector OR [B, d] per-graph vectors
            batch_size: Required for fixed batch mode, None for variable batch
            x_cell_embedded: [N, d] embedded original input for injection
        """
        x_dict = {'cell': x_cell}

        x1 = self.hconv1(x_dict, edge_index_dict)['cell']
        x1 = self._res_block(x_cell, x1, self.norm1)

        x2 = self.hconv2({'cell': x1}, edge_index_dict)['cell']
        x2 = self._res_block(x1, x2, self.norm2)

        x3 = self.hgt({'cell': x2}, edge_index_dict)['cell']
        x3 = self._res_block(x2, x3, self.norm_hgt)

        if self.use_input_injection and x_cell_embedded is not None:
            x3 = x3 + self.input_proj(x_cell_embedded)

        x3 = self.film(x3, z_h, batch_size=batch_size)
        return x3


class _HModule(nn.Module):
    """
    H-module: maintains a global state z_H per graph.
    Updates z_H using attention pooling over node states.
    Requires fixed batch size (uniform nodes per graph).
    """
    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = _RMSNorm(hidden_dim)

    def forward(self, nodes, z_prev, batch_size: int, return_attention: bool = False):
        B = batch_size
        C = nodes.shape[0] // B
        d = self.hidden_dim
        H = self.num_heads
        Dh = self.head_dim

        nodes = nodes.view(B, C, d)

        q = self.q(z_prev).view(B, H, Dh)
        k = self.k(nodes).view(B, C, H, Dh).permute(0, 2, 1, 3)
        v = self.v(nodes).view(B, C, H, Dh).permute(0, 2, 1, 3)

        scores = torch.einsum("bhd,bhcd->bhc", q, k) / (Dh ** 0.5)
        w = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhc,bhcd->bhd", w, v).contiguous()
        ctx = ctx.view(B, d)

        z_in = torch.cat([ctx, z_prev], dim=-1)
        z_out = self.norm(z_prev + self.mlp(z_in))

        if return_attention:
            w_mean = w.mean(dim=1)
            return z_out, w_mean

        return z_out


class _Readout(nn.Module):
    """Produces per-node logits conditioned on global z_H."""
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, nodes, z_h, batch_size: int = None):
        """
        Args:
            nodes: [N, d] node features
            z_h: [d] shared vector OR [B, d] per-graph vectors
            batch_size: Required for fixed batch mode, None for variable batch
        """
        N, d = nodes.shape

        if z_h.dim() == 1:
            # Shared z: expand and concat
            z_expanded = z_h.unsqueeze(0).expand(N, -1)  # [N, d]
            return self.mlp(torch.cat([nodes, z_expanded], dim=-1)).squeeze(-1)

        # Per-graph z: requires fixed batch
        B = batch_size
        assert B is not None, "batch_size required for per-graph z"
        assert N % B == 0, "Fixed batch mode requires uniform nodes per graph"
        C = N // B

        nodes_ = nodes.view(B, C, d)
        z_ = z_h[:, None, :].expand(B, C, d)
        logits = self.mlp(torch.cat([nodes_, z_], dim=-1)).squeeze(-1)
        return logits.reshape(N)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model with L-module (fast local) and H-module (slow global).
    
    Args:
        same_size_batches: If True, uses fixed batch mode (reshape-based, slightly faster).
                          If False, uses variable batch mode (broadcast-based).
        use_hmod: If True, uses H-module for global reasoning (requires same_size_batches=True).
                  If False, uses learned z per cycle.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gat_heads: int = 2,
        hgt_heads: int = 4,
        dropout: float = 0.10,
        use_batch_norm: bool = True,
        n_cycles: int = 3,
        t_micro: int = 2,
        use_input_injection: bool = True,
        z_dim: int = 256,
        use_hmod: bool = False,
        same_size_batches: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.t_micro = t_micro
        self.use_input_injection = use_input_injection
        self.use_hmod = use_hmod
        self.same_size_batches = same_size_batches

        if use_hmod and not same_size_batches:
            raise ValueError("use_hmod=True requires same_size_batches=True")

        self.edge_types = [
            ('cell', 'line_constraint', 'cell'),
            ('cell', 'region_constraint', 'cell'),
            ('cell', 'diagonal_constraint', 'cell')
        ]

        print(f"HRM configured:")
        print(f"Cycles: {n_cycles}, Micro-steps: {t_micro}")
        print(f"GAT heads: {gat_heads}, HGT heads: {hgt_heads}")
        print(f"same_size_batches: {same_size_batches}")
        print(f"use_hmod: {use_hmod}")

        if use_hmod:
            print(f"Mode: H-module (per-graph z, attention pooling)")
        else:
            print(f"Mode: z-per-cycle (shared z)")

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

        self.readout = _Readout(hidden_dim, dropout)

        if self.use_hmod:
            self.h_mod = _HModule(hidden_dim=hidden_dim, dropout=dropout, num_heads=hgt_heads)
            self.z0 = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.z_per_cycle = nn.ParameterList([
                nn.Parameter(torch.randn(z_dim) * 0.02)
                for _ in range(n_cycles)
            ])
            self.z_proj = nn.Linear(z_dim, hidden_dim)

    def forward(self, batch, return_intermediates=False):
        """
        Forward pass.
        
        When same_size_batches=True: all graphs must have same node count
        When same_size_batches=False: variable graph sizes allowed (z-per-cycle only)
        """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        B = batch.num_graphs

        x_in = x_dict['cell']
        nodes_embedded = self.embed(x_in)
        nodes = nodes_embedded

        # batch_size arg: pass B for fixed batch mode, None for variable
        bs_arg = B if self.same_size_batches else None

        if return_intermediates:
            L_states = []

        if self.use_hmod:
            z = self.z0[None, :].expand(B, -1)  # [B, d]
            
            for cycle_idx in range(self.n_cycles):
                for micro_idx in range(self.t_micro):
                    nodes = self.l_block(
                        nodes, edge_index_dict, z,
                        batch_size=B,
                        x_cell_embedded=nodes_embedded if self.use_input_injection else None
                    )
                    if return_intermediates:
                        L_states.append(nodes.detach().cpu())

                z = self.h_mod(nodes, z, batch_size=B)

            logits = self.readout(nodes, z, batch_size=B)

        else:
            for cycle_idx in range(self.n_cycles):
                z = self.z_proj(self.z_per_cycle[cycle_idx])  # [d] shared
                
                for micro_idx in range(self.t_micro):
                    nodes = self.l_block(
                        nodes, edge_index_dict, z,
                        batch_size=bs_arg,
                        x_cell_embedded=nodes_embedded if self.use_input_injection else None
                    )
                    if return_intermediates:
                        L_states.append(nodes.detach().cpu())

            logits = self.readout(nodes, z, batch_size=bs_arg)

        if return_intermediates:
            batch_indices = batch['cell'].batch
            first_graph_nodes = (batch_indices == 0).sum().item()
            board_size = int(first_graph_nodes ** 0.5)
            
            intermediates = {
                'L_states': L_states,
                'final_logits': logits.detach().cpu(),
                'board_size': board_size
            }
            return logits, intermediates

        return logits