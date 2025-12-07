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

    def forward(self, x_dict, edge_index_dict, batch=None):
        """Forward pass with multi-stage global context injection for Type 2 error prevention.

        Args:
            x_dict: dict with 'cell' key containing node features
            edge_index_dict: dict of edge indices per constraint type
            batch: unused, for API compatibility with HRM
        """
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
    """Feature-wise Linear Modulation: conditions node features on global z_H.

    Supports dual z_H mode where both z_local and z_meta are concatenated for conditioning.
    """
    def __init__(self, hidden_dim: int, use_dual_z: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dual_z = use_dual_z

        input_dim = 2 * hidden_dim if use_dual_z else hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        )

    def forward(self, nodes, z_local, z_meta=None, batch=None):
        """
        nodes: [C, d] node features
        z_local: [B, d] per-graph global vectors (or [d] for single graph)
        z_meta: [d] batch-wide global vector (only used if use_dual_z=True)
        batch: [C] graph assignment for each node (None if single graph)
        """
        if batch is None:
            # Single graph case (inference or batch_size=1)
            if self.use_dual_z:
                # z_meta should equal z_local for single graph, but concat anyway for consistency
                z_combined = torch.cat([z_local, z_meta if z_meta is not None else z_local], dim=-1)
            else:
                z_combined = z_local
            gamma, beta = self.mlp(z_combined).chunk(2, dim=-1)
            return gamma * nodes + beta
        else:
            # Batched case
            if self.use_dual_z:
                # z_local: [B, d], z_meta: [d]
                # Expand z_meta to [B, d] and concatenate
                B = z_local.shape[0]
                z_meta_expanded = z_meta.unsqueeze(0).expand(B, -1)  # [B, d]
                z_combined = torch.cat([z_local, z_meta_expanded], dim=-1)  # [B, 2d]
            else:
                z_combined = z_local  # [B, d]

            film_out = self.mlp(z_combined)  # [B, 2d]
            gamma, beta = film_out.chunk(2, dim=-1)  # [B, d] each
            gamma_per_node = gamma[batch]  # [C, d]
            beta_per_node = beta[batch]    # [C, d]
            return gamma_per_node * nodes + beta_per_node


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
        use_dual_z: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2)
        self.use_bn = use_batch_norm
        self.use_input_injection = use_input_injection
        self.use_dual_z = use_dual_z

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

        self.film = _FiLM(hidden_dim, use_dual_z=use_dual_z)

    def _res_block(self, x_old, x_new, bn):
        if bn is not None:
            x_new = bn(x_new)
        return self.relu(x_old + self.drop(x_new))

    def forward(self, x_cell, edge_index_dict, z_local, z_meta=None, x_cell_embedded=None, batch=None):
        """
        x_cell: [C, d] current node states
        z_local: [B, d] per-graph global vectors (or [d] for single graph)
        z_meta: [d] batch-wide global vector (only used if use_dual_z=True)
        x_cell_embedded: [C, d] embedded original input for injection
        batch: [C] graph assignment for each node (None if single graph)
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

        x3 = self.film(x3, z_local, z_meta=z_meta, batch=batch)
        return x3


class _HModule(nn.Module):
    """H-module: updates global state z_H via multi-head attention pooling over nodes.

    Supports dual z_H mode where both per-graph (z_local) and batch-wide (z_meta) contexts are computed.
    """
    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4, use_dual_z: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_dual_z = use_dual_z

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Shared projections for keys/values
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Separate query projections for local and meta (if dual mode)
        self.query_local = nn.Linear(hidden_dim, hidden_dim)
        if use_dual_z:
            self.query_meta = nn.Linear(hidden_dim, hidden_dim)
            self.mlp_meta = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm_meta = _RMSNorm(hidden_dim)

        self.mlp_local = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_local = _RMSNorm(hidden_dim)

        # Backward compatibility alias
        self.query = self.query_local
        self.mlp = self.mlp_local
        self.norm = self.norm_local

    def forward(self, nodes, z_prev_local, z_prev_meta=None, batch=None, num_graphs=None, return_attention=False):
        """
        Compute z_local (per-graph) and optionally z_meta (batch-wide).

        Args:
            nodes: [C, d] all node states
            z_prev_local: [B, d] per-graph previous states (or [d] for single graph)
            z_prev_meta: [d] batch-wide previous state (only used if use_dual_z=True)
            batch: [C] graph assignment for each node (None if single graph)
            num_graphs: int, number of graphs in batch (required if batch is provided)

        Returns:
            If use_dual_z=False: z_local (and optionally attention)
            If use_dual_z=True: (z_local, z_meta) tuple (and optionally attention)
        """
        C, d = nodes.shape
        H = self.num_heads
        hd = self.head_dim

        # Project keys and values (shared for both local and meta)
        K = self.key(nodes).view(C, H, hd)    # [C, H, d/H]
        V = self.value(nodes).view(C, H, hd)  # [C, H, d/H]

        if batch is None:
            # Single graph case - z_local and z_meta are the same
            Q = self.query_local(z_prev_local).view(1, H, hd)  # [1, H, d/H]

            attn_scores = torch.einsum('qhd,chd->hc', Q, K) / (hd ** 0.5)  # [H, C]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [H, C]

            pooled_heads = torch.einsum('hc,chd->hd', attn_weights, V)  # [H, d/H]
            pooled = pooled_heads.reshape(self.hidden_dim)  # [d]

            z_local = torch.cat([pooled, z_prev_local], dim=-1)  # [2d]
            z_local_out = self.norm_local(self.mlp_local(z_local))  # [d]

            if self.use_dual_z:
                # For single graph, z_meta = z_local
                if return_attention:
                    return z_local_out, z_local_out, attn_weights.mean(dim=0)
                return z_local_out, z_local_out
            else:
                if return_attention:
                    return z_local_out, attn_weights.mean(dim=0)
                return z_local_out
        else:
            # Batched case
            B = num_graphs

            # Compute z_local (per-graph pooling)
            Q_local = self.query_local(z_prev_local).view(B, H, hd)  # [B, H, d/H]
            Q_local_per_node = Q_local[batch]  # [C, H, d/H]
            attn_scores_local = (Q_local_per_node * K).sum(dim=-1) / (hd ** 0.5)  # [C, H]
            attn_weights_local = pyg_softmax(attn_scores_local, batch, num_nodes=C)  # [C, H]

            weighted_V_local = attn_weights_local.unsqueeze(-1) * V  # [C, H, d/H]
            batch_expanded = batch.view(C, 1, 1).expand(-1, H, hd)
            pooled_local = torch.zeros(B, H, hd, device=nodes.device, dtype=nodes.dtype)
            pooled_local.scatter_add_(0, batch_expanded, weighted_V_local)  # [B, H, d/H]
            pooled_local = pooled_local.reshape(B, self.hidden_dim)  # [B, d]

            z_local = torch.cat([pooled_local, z_prev_local], dim=-1)  # [B, 2d]
            z_local_out = self.norm_local(self.mlp_local(z_local))  # [B, d]

            if self.use_dual_z:
                # Compute z_meta (batch-wide pooling)
                Q_meta = self.query_meta(z_prev_meta).view(1, H, hd)  # [1, H, d/H]

                attn_scores_meta = torch.einsum('qhd,chd->hc', Q_meta, K) / (hd ** 0.5)  # [H, C]
                attn_weights_meta = torch.softmax(attn_scores_meta, dim=-1)  # [H, C]

                pooled_heads_meta = torch.einsum('hc,chd->hd', attn_weights_meta, V)  # [H, d/H]
                pooled_meta = pooled_heads_meta.reshape(self.hidden_dim)  # [d]

                z_meta = torch.cat([pooled_meta, z_prev_meta], dim=-1)  # [2d]
                z_meta_out = self.norm_meta(self.mlp_meta(z_meta))  # [d]

                if return_attention:
                    return z_local_out, z_meta_out, attn_weights_local.mean(dim=1)
                return z_local_out, z_meta_out
            else:
                if return_attention:
                    return z_local_out, attn_weights_local.mean(dim=1)
                return z_local_out


class _Readout(nn.Module):
    """Produces per-cell logits conditioned on global z_H.

    Supports dual z_H mode where both z_local and z_meta are concatenated for conditioning.
    """
    def __init__(self, hidden_dim: int, dropout: float, use_dual_z: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dual_z = use_dual_z

        # Input: nodes [d] + z_local [d] + (z_meta [d] if dual)
        # So input dim is 2d or 3d
        input_dim = 3 * hidden_dim if use_dual_z else 2 * hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, nodes, z_local, z_meta=None, batch=None):
        """
        nodes: [C, d] node features
        z_local: [B, d] per-graph global vectors (or [d] for single graph)
        z_meta: [d] batch-wide global vector (only used if use_dual_z=True)
        batch: [C] graph assignment for each node (None if single graph)
        """
        C = nodes.shape[0]

        if batch is None:
            # Single graph case
            z_local_tiled = z_local.unsqueeze(0).expand_as(nodes)  # [C, d]
            if self.use_dual_z:
                z_meta_tiled = (z_meta if z_meta is not None else z_local).unsqueeze(0).expand_as(nodes)
                combined = torch.cat([nodes, z_local_tiled, z_meta_tiled], dim=-1)  # [C, 3d]
            else:
                combined = torch.cat([nodes, z_local_tiled], dim=-1)  # [C, 2d]
        else:
            # Batched case
            z_local_tiled = z_local[batch]  # [C, d]
            if self.use_dual_z:
                z_meta_tiled = z_meta.unsqueeze(0).expand(C, -1)  # [C, d]
                combined = torch.cat([nodes, z_local_tiled, z_meta_tiled], dim=-1)  # [C, 3d]
            else:
                combined = torch.cat([nodes, z_local_tiled], dim=-1)  # [C, 2d]

        return self.mlp(combined).squeeze(-1)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model with L-module (fast local) and H-module (slow global).
    Based on Wang et al., 2025.

    Supports dual z_H mode: combines per-graph z_local (puzzle-specific reasoning)
    with batch-wide z_meta (cross-puzzle pattern learning).
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
        use_dual_z: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.t_micro = t_micro
        self.use_input_injection = use_input_injection
        self.use_dual_z = use_dual_z

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
        print(f"Dual z_H (local + meta): {use_dual_z}")

        self.embed = _InitialEmbed(input_dim, hidden_dim, dropout)
        self.l_block = _LBlock(
            hidden_dim=hidden_dim,
            gat_heads=gat_heads,
            hgt_heads=hgt_heads,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            edge_types=self.edge_types,
            use_input_injection=use_input_injection,
            use_dual_z=use_dual_z,
        )
        self.h_mod = _HModule(hidden_dim, dropout, num_heads=h_pooling_heads, use_dual_z=use_dual_z)
        self.readout = _Readout(hidden_dim, dropout, use_dual_z=use_dual_z)

        if z_init == "learned":
            self.z0 = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        else:
            self.register_buffer("z0", torch.zeros(hidden_dim), persistent=False)

    def forward(self, x_dict, edge_index_dict, batch=None, return_intermediates=False):
        """
        Forward pass through hierarchical L/H architecture.

        Args:
            x_dict: dict with 'cell' key containing node features [C, input_dim]
            edge_index_dict: dict of edge indices per constraint type
            batch: [C] tensor with graph assignment for each node (None for single graph)
            return_intermediates: if True, return L_states for visualization

        Returns:
            logits [C] or (logits, intermediates) if return_intermediates=True.
        """
        x_in = x_dict['cell']
        nodes_embedded = self.embed(x_in)
        nodes = nodes_embedded

        # Determine batch size and initialize z_H
        if batch is None:
            # Single graph (inference or batch_size=1)
            z_local = self.z0  # [d]
            z_meta = self.z0 if self.use_dual_z else None  # [d]
            num_graphs = None
        else:
            # Batched training
            num_graphs = batch.max().item() + 1
            z_local = self.z0.unsqueeze(0).expand(num_graphs, -1).clone()  # [B, d]
            z_meta = self.z0.clone() if self.use_dual_z else None  # [d]

        if return_intermediates:
            L_states = []

        for cycle_idx in range(self.n_cycles):
            for micro_idx in range(self.t_micro):
                nodes = self.l_block(
                    nodes,
                    edge_index_dict,
                    z_local,
                    z_meta=z_meta,
                    x_cell_embedded=nodes_embedded if self.use_input_injection else None,
                    batch=batch
                )

                if return_intermediates:
                    L_states.append(nodes.detach().cpu())

            # Update z_H via H-module
            if self.use_dual_z:
                z_local, z_meta = self.h_mod(
                    nodes, z_local, z_prev_meta=z_meta,
                    batch=batch, num_graphs=num_graphs, return_attention=False
                )
            else:
                z_local = self.h_mod(
                    nodes, z_local, batch=batch, num_graphs=num_graphs, return_attention=False
                )

        logits = self.readout(nodes, z_local, z_meta=z_meta, batch=batch)

        if return_intermediates:
            intermediates = {
                'L_states': L_states,  # List of tensors (t_micro * n_cycles)
                'board_size': int(x_in.shape[0] ** 0.5) if batch is None else None
            }
            return logits, intermediates

        return logits
