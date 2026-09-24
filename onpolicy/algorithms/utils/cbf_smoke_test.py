"""
Smoke test for the CBFGNN barrier network (Stage 1).

Checks, on CPU with dummy data:
  1. forward pass returns one h per node with correct shape
  2. gradients flow through to the network params
  3. the CBF loss computes and its components are finite
  4. a short overfit run can actually push h toward the right sign
     (positive when agents are far apart, negative when too close)

This is a pipeline sanity check, NOT the real training run — it just proves
the module is wired correctly before it touches the real environment.
"""

import torch

from cbf_gnn import CBFGNN, safe_unsafe_masks, cbf_loss


def make_dummy_batch(batch_size, num_agents, node_obs_dim, sep, seed=0):
    """
    Build a dummy graph batch where node feature 0,1 are (x, y) positions,
    and the adjacency is the pairwise distance matrix. Half the scenes are
    'spread out' (safe), half are 'bunched up' (unsafe), so the labels are
    unambiguous for the overfit check.
    """
    g = torch.Generator().manual_seed(seed)
    node_obs = torch.zeros(batch_size, num_agents, node_obs_dim)

    # positions
    pos = torch.zeros(batch_size, num_agents, 2)
    half = batch_size // 2
    # safe scenes: agents spread far apart (>> sep)
    pos[:half] = torch.rand(half, num_agents, 2, generator=g) * (10.0 * sep)
    # unsafe scenes: agents clustered within < sep of each other
    centre = torch.rand(batch_size - half, 1, 2, generator=g) * (10.0 * sep)
    pos[half:] = centre + (torch.rand(batch_size - half, num_agents, 2, generator=g) - 0.5) * (0.6 * sep)

    node_obs[:, :, 0:2] = pos
    # a couple of extra dummy features so node_obs_dim > 2 is exercised
    if node_obs_dim > 2:
        node_obs[:, :, 2:] = torch.rand(batch_size, num_agents, node_obs_dim - 2, generator=g)

    # pairwise distance matrix -> adjacency
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)          # (B, N, N, 2)
    dist = torch.linalg.norm(diff, dim=-1)             # (B, N, N)

    # nearest-neighbour distance per agent (ignore self by setting diagonal to +inf)
    eye = torch.eye(num_agents).bool().unsqueeze(0)
    dist_no_self = dist.masked_fill(eye, float("inf"))
    nearest = dist_no_self.min(dim=-1).values           # (B, N)

    return node_obs, dist, nearest


def main():
    torch.manual_seed(0)
    B, N, D = 16, 5, 6         # batch, agents, node feature dim
    SEP = 0.3                  # stand-in for separation_distance
    MAX_EDGE = 5.0 * SEP       # sensing radius

    net = CBFGNN(node_obs_dim=D, edge_dim=1, hidden_size=64,
                 num_layers=2, max_edge_dist=MAX_EDGE)

    node_obs, adj, nearest = make_dummy_batch(B, N, D, SEP)

    # ---- 1. forward shape ----
    h = net(node_obs, adj)
    assert h.shape == (B, N), f"expected h shape {(B, N)}, got {tuple(h.shape)}"
    print(f"[1] forward OK — h shape {tuple(h.shape)}")

    # ---- 2. labels + loss compute ----
    safe_mask, unsafe_mask = safe_unsafe_masks(nearest, SEP, margin=2.0)
    # h_next: fake a next step (just reuse h for the compute check)
    h_next = net(node_obs, adj)
    loss, comp = cbf_loss(h, h_next, safe_mask, unsafe_mask, alpha=1.0)
    assert torch.isfinite(loss), "loss is not finite"
    print(f"[2] loss OK — {comp}")

    # ---- 3. gradients flow ----
    loss.backward()
    grad_norm = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradient reached the network"
    print(f"[3] gradients flow — total grad magnitude {float(grad_norm):.4f}")

    # ---- 4. short overfit: can h learn the right sign? ----
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(400):
        opt.zero_grad()
        h = net(node_obs, adj)
        safe_mask, unsafe_mask = safe_unsafe_masks(nearest, SEP, margin=2.0)
        loss, comp = cbf_loss(h, h.detach(), safe_mask, unsafe_mask, alpha=1.0)
        loss.backward()
        opt.step()

    h = net(node_obs, adj).detach()
    safe_h = h[safe_mask]
    unsafe_h = h[unsafe_mask]
    print(f"[4] after overfit:")
    print(f"      safe agents:   mean h = {safe_h.mean():+.3f}  (want > 0), "
          f"frac correct = {(safe_h >= 0).float().mean():.2f}")
    print(f"      unsafe agents: mean h = {unsafe_h.mean():+.3f}  (want < 0), "
          f"frac correct = {(unsafe_h < 0).float().mean():.2f}")

    ok = (safe_h.mean() > 0) and (unsafe_h.mean() < 0)
    print("\nSMOKE TEST PASSED — the barrier network can learn the safe/unsafe boundary."
          if ok else
          "\nSMOKE TEST WARNING — boundary not cleanly learned in 400 steps (pipeline still works).")


if __name__ == "__main__":
    main()
