"""
CBFGNN — Graph Control Barrier Function network.

A small graph neural network that reads the SAME graph representation the policy
already uses (node_obs + distance adjacency) and outputs a single scalar barrier
value h per agent:

    h > 0  ->  agent is in the safe region
    h ~ 0  ->  agent is on the safety boundary
    h < 0  ->  agent is in the unsafe region (separation violated)

This mirrors the input handling in onpolicy/algorithms/utils/gnn_new.py so the
barrier network sees graphs identically to the actor/critic. It deliberately
does NOT touch the policy — it is a standalone module trained with its own loss
(see cbf_loss below).

Stage 1 of the Option B safety-layer plan.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import TransformerConv


def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
    """
    Convert a distance-magnitude adjacency matrix into (edge_index, edge_attr).

    This replicates GNNBase.process_adj from gnn_new.py so the barrier network
    builds edges the same way the policy does. Nodes closer than `max_edge_dist`
    (and not self-connected) become edges; the distance becomes the edge weight.

    adj: (batch_size, num_nodes, num_nodes) OR (num_nodes, num_nodes)
    returns:
        edge_index: (2, num_edges)
        edge_attr:  (num_edges, 1)
    """
    assert adj.dim() in (2, 3)
    assert adj.size(-1) == adj.size(-2)

    connect_mask = ((adj < max_edge_dist) & (adj > 0)).float()
    adj = adj * connect_mask

    if adj.dim() == 3:
        batch_size, num_nodes, _ = adj.shape
        edge_index = adj.nonzero(as_tuple=False)
        edge_attr = adj[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
        batch = edge_index[:, 0] * num_nodes
        edge_index = torch.stack(
            [batch + edge_index[:, 1], batch + edge_index[:, 2]], dim=0
        )
    else:
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        edge_attr = adj[edge_index[0], edge_index[1]]

    edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr
    return edge_index, edge_attr


class CBFGNN(nn.Module):
    """
    Barrier-value network.

    Args:
        node_obs_dim: width of each node's feature vector (node_obs last dim).
        edge_dim:     edge feature width (1 for the distance weight).
        hidden_size:  hidden width of the GNN / MLP layers.
        num_layers:   number of TransformerConv message-passing layers.
        max_edge_dist: sensing radius used to build edges from the distance adj.
        num_heads:    attention heads for TransformerConv.
    """

    def __init__(
        self,
        node_obs_dim: int,
        edge_dim: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        max_edge_dist: float = 1.0,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.max_edge_dist = max_edge_dist

        # Input encoder: node features -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(node_obs_dim, hidden_size),
            nn.ReLU(),
        )

        # Message-passing layers (graph attention), matching the policy's family
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    heads=num_heads,
                    concat=False,      # keep width == hidden_size for stacking
                    beta=False,
                    dropout=0.0,
                    edge_dim=edge_dim,
                    root_weight=True,
                )
            )
        self.activation = nn.ReLU()

        # Head: node embedding -> scalar barrier value h
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, node_obs: Tensor, adj: Tensor) -> Tensor:
        """
        node_obs: (batch_size, num_nodes, node_obs_dim)
        adj:      (batch_size, num_nodes, num_nodes)  distance-magnitude matrix

        returns:
            h: (batch_size, num_nodes) barrier value for every node.
               Callers pick out the agent nodes they care about.
        """
        batch_size, num_nodes, _ = node_obs.shape

        # Flatten batch of graphs into one big disconnected graph for PyG.
        x = node_obs.reshape(batch_size * num_nodes, -1)
        edge_index, edge_attr = process_adj(adj, self.max_edge_dist)

        x = self.encoder(x)
        for conv in self.convs:
            x = self.activation(conv(x, edge_index, edge_attr))

        h = self.head(x)                       # (batch*nodes, 1)
        h = h.reshape(batch_size, num_nodes)   # (batch, nodes)
        return h


# ---------------------------------------------------------------------------
# Safe / unsafe labelling and the CBF loss
# ---------------------------------------------------------------------------

def safe_unsafe_masks(
    agent_dist: Tensor,
    separation_distance: float,
    margin: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Build per-agent safe / unsafe labels from inter-agent distances.

    agent_dist: (batch, num_agents) — each agent's distance to its NEAREST
                other agent (min over neighbours). This is what decides safety.
    separation_distance: the minimum allowed separation (unsafe below this).
    margin: multiplier defining a clearly-safe band. An agent is labelled
            "safe" only when its nearest neighbour is comfortably far
            (> margin * separation_distance); "unsafe" when closer than the
            separation distance. The gap between is left unlabelled so the
            boundary can form there.

    returns:
        safe_mask:   (batch, num_agents) bool
        unsafe_mask: (batch, num_agents) bool
    """
    unsafe_mask = agent_dist < separation_distance
    safe_mask = agent_dist > (margin * separation_distance)
    return safe_mask, unsafe_mask


def cbf_loss(
    h: Tensor,
    h_next: Tensor,
    safe_mask: Tensor,
    unsafe_mask: Tensor,
    alpha: float = 1.0,
    eps: float = 0.02,
    w_safe: float = 1.0,
    w_unsafe: float = 1.0,
    w_hdot: float = 0.1,
) -> Tuple[Tensor, dict]:
    """
    The three-term CBF loss (GCBF-style).

    h:          (batch, num_agents) barrier value at the current step.
    h_next:     (batch, num_agents) barrier value at the NEXT step (same agents),
                used to estimate the time-derivative h_dot = (h_next - h)/dt.
                Pass with dt already folded in, or leave dt=1 (per-step).
    safe_mask / unsafe_mask: from safe_unsafe_masks().
    alpha:  class-K gain in the barrier condition  h_dot + alpha*h >= 0.
    eps:    small strict-inequality margin.

    Loss terms:
      - safe:   h should be >= 0 in the safe region      -> relu(-h + eps)
      - unsafe: h should be <  0 in the unsafe region    -> relu(h + eps)
      - h_dot:  barrier condition h_dot + alpha*h >= 0    -> relu(-(h_dot) - alpha*h + eps)

    returns (total_loss, components_dict_for_logging)
    """
    h_dot = h_next - h  # per-step derivative estimate (dt = 1)

    # Safe region: want h >= 0
    if safe_mask.any():
        loss_safe = torch.relu(-h[safe_mask] + eps).mean()
        acc_safe = (h[safe_mask] >= 0).float().mean()
    else:
        loss_safe = h.sum() * 0.0
        acc_safe = torch.tensor(1.0, device=h.device)

    # Unsafe region: want h < 0
    if unsafe_mask.any():
        loss_unsafe = torch.relu(h[unsafe_mask] + eps).mean()
        acc_unsafe = (h[unsafe_mask] < 0).float().mean()
    else:
        loss_unsafe = h.sum() * 0.0
        acc_unsafe = torch.tensor(1.0, device=h.device)

    # Barrier (derivative) condition everywhere: h_dot + alpha*h >= 0
    loss_hdot = torch.relu(-(h_dot) - alpha * h + eps).mean()

    total = w_safe * loss_safe + w_unsafe * loss_unsafe + w_hdot * loss_hdot

    components = {
        "loss_total": float(total.detach()),
        "loss_safe": float(loss_safe.detach()),
        "loss_unsafe": float(loss_unsafe.detach()),
        "loss_hdot": float(loss_hdot.detach()),
        "acc_safe": float(acc_safe.detach()),
        "acc_unsafe": float(acc_unsafe.detach()),
    }
    return total, components
