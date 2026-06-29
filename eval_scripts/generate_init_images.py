#!/usr/bin/env python
"""
Generate 5 PNG images showing random corridor-scenario initializations
for 5 agents (working_three_phase_graph scenario).

Usage:
    python eval_scripts/generate_init_images.py
    python eval_scripts/generate_init_images.py --output_dir /tmp/init
"""
import argparse
import os
import sys
from typing import Optional

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]

from multiagent.custom_scenarios.working_three_phase_graph import Scenario  # noqa: E402


class Args:
    num_agents: int = 5
    world_size: float = 8.0  # match actual eval/training value
    num_scripted_agents: int = 0
    num_obstacles: int = 0
    collaborative: bool = False
    max_speed: Optional[float] = 2.0
    collision_rew: float = 5.0
    formation_rew: float = 2.0
    goal_rew: float = 50.0
    min_dist_thresh: float = 0.1
    use_dones: bool = True
    episode_length: int = 450
    max_edge_dist: float = 5.0
    graph_feat_type: str = "relative"
    fair_wt: float = 1.0
    fair_rew: float = 1.0
    formation_type: str = "point"
    dynamics_type: str = "unicycle_vehicle"
    num_landmarks: int = 5
    num_walls: int = 0
    total_actions: int = 5
    n_rollout_threads: int = 1
    num_env_steps: int = 10_000_000
    render_episodes: int = 5
    zeroshift: bool = False


AGENT_COLORS = ["#E84040", "#40C840", "#4040E8", "#C8B820", "#20C8C8"]
GOAL_COLORS = ["#FF8080", "#80FF80", "#8080FF", "#E8E020", "#40E8E8"]


def compute_bounds(world, pad_frac=0.20):
    """Auto-compute tight bounds containing corridor, agents, and goals."""
    all_pts = []

    tp = world.tube_params
    entrance = tp["entrance"]
    exit_pt = tp["exit"]
    hw = tp["half_width"]
    n = tp["n"]
    for pt in [entrance, exit_pt]:
        all_pts.append(pt + hw * n)
        all_pts.append(pt - hw * n)

    for agent in world.agents:
        all_pts.append(agent.state.p_pos)

    for lm in world.landmarks:
        all_pts.append(lm.state.p_pos)

    pts = np.array(all_pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = span * pad_frac
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def draw_corridor(ax, tube_params):
    entrance = tube_params["entrance"]
    exit_pt = tube_params["exit"]
    half_w = tube_params["half_width"]
    n = tube_params["n"]

    corners = [
        entrance + half_w * n,
        entrance - half_w * n,
        exit_pt - half_w * n,
        exit_pt + half_w * n,
    ]
    ax.add_patch(patches.Polygon(
        corners, closed=True,
        facecolor="white", edgecolor="#888888",
        linewidth=1.2, linestyle="--", zorder=1,
    ))


def draw_agent_triangle(ax, pos, heading, color, span=1.0):
    """Oriented triangle scaled to 1.2% of scene span."""
    size = span * 0.012
    tri = np.array([
        [size, 0.0],
        [-size * 0.6, size * 0.55],
        [-size * 0.6, -size * 0.55],
    ])
    rot = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)],
    ])
    tri_rot = tri @ rot.T + pos
    ax.add_patch(patches.Polygon(
        tri_rot, closed=True,
        facecolor=color, edgecolor="black", linewidth=0.6, zorder=6,
    ))


def render_init(scenario, world, idx, output_dir, seed):
    xmin, xmax, ymin, ymax = compute_bounds(world)
    width = xmax - xmin
    height = ymax - ymin
    span = max(width, height)

    fig_w = 5.0
    fig_h = min(fig_w * (height / max(width, 1e-3)), 9.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    draw_corridor(ax, world.tube_params)

    for i, agent in enumerate(world.agents):
        draw_agent_triangle(
            ax, agent.state.p_pos, float(agent.state.theta),
            AGENT_COLORS[i % len(AGENT_COLORS)],
            span=span,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    angle_deg = float(np.degrees(world.tube_params["angle"]))
    fname = os.path.join(
        output_dir, f"init_{idx + 1:02d}_angle{angle_deg:+.0f}deg.png"
    )
    fig.savefig(fname, dpi=120, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="eval_scripts/init_images")
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    cli = parser.parse_args()

    os.makedirs(cli.output_dir, exist_ok=True)

    args = Args()
    scenario = Scenario()
    world = scenario.make_world(args)

    print(f"Generating {cli.n_images} images -> {cli.output_dir}/")
    print(
        f"  sep_dist={scenario.separation_distance:.3f} km"
        f"  world_size={scenario.world_size}"
    )
    saved = []
    for i in range(cli.n_images):
        seed = cli.base_seed + i
        np.random.seed(seed)
        # Use full curriculum (ratio=1.0) for maximum angle variety
        scenario.reset_world(world, num_current_episode=args.render_episodes)
        saved.append(render_init(scenario, world, i, cli.output_dir, seed))

    print(f"\nDone. {len(saved)} images written to {cli.output_dir}/")


if __name__ == "__main__":
    main()
