#!/usr/bin/env python
import argparse
from distutils.util import strtobool 
import sys
import os
from typing import Dict
import wandb
import numpy as np
from pathlib import Path
import torch

import os,sys
sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box
from onpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, GraphSubprocVecEnv, GraphDummyVecEnv

def make_render_env(all_args:argparse.Namespace):
    def get_env_fn(rank:int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)

            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv([get_env_fn(i) for i in 
                                        range(all_args.n_rollout_threads)])
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=2)
    parser.add_argument('--num_agents', type=int, default=2, 
                        help="number of players")
    parser.add_argument('--num_obstacles', type=int, default=0, 
                        help="Number of obstacles")
    parser.add_argument("--collaborative", type=lambda x:bool(strtobool(x)), 
                    default=False, help="Number of agents in the env")
    parser.add_argument("--max_speed", type=float, default=2, 
                        help="Max speed for agents. NOTE that if this is None, "
                        "then max_speed is 2 with discrete action space")
    # parser.add_argument("--goal_rew", type=float, default=20, 
    #                     help="The reward to be added if agent reaches the goal")
    # parser.add_argument("--fair_wt", type=int,
    #                     default=1,help="choice of weight")
    # parser.add_argument("--fair_rew", type=int,
    #                     default=1, help ="choice of fairness reward")
    parser.add_argument("--min_dist_thresh", type=float, default=0.05, 
                        help="The minimum distance threshold to classify whether "
                        "agent has reached the goal or not")
    parser.add_argument("--use_dones", type=lambda x:bool(strtobool(x)),
                        default=True, help="Whether we want to use the 'done=True' "
                        "when agent has reached the goal or just return False like "
                        "the `simple.py` or `simple_spread.py`")
    '''
    parser.add_argument("--dynamics_type", type=str,
                        default="unicycle_vehicle",
                        choices=["unicycle_vehicle", "double_integrator", "air_taxi"],
                        help="Dynamics model type for agents.")
    '''

    # === CBF evaluation (added) — all off by default, so a plain invocation of
    # eval_mpe.py behaves exactly as the original. When --cbf_eval is set, the
    # runner's render() loop additionally records separation-severity metrics
    # (read-only); when --cbf_filter is also set, the CBF safety filter masks
    # unsafe primitives in-loop. The original metrics are unchanged either way. ===
    parser.add_argument("--cbf_eval", action="store_true",
                        help="record the extra CBF safety/severity metrics in render()")
    parser.add_argument("--cbf_filter", action="store_true",
                        help="apply the CBF safety filter in render() (needs --cbf_barrier)")
    parser.add_argument("--cbf_barrier", type=str, default="cbf_stage1_barrier.pt")
    parser.add_argument("--cbf_alpha", type=float, default=1.0)
    parser.add_argument("--cbf_eps", type=float, default=0.0)
    parser.add_argument("--cbf_horizon", type=int, default=3)
    parser.add_argument("--cbf_fallback", type=str, default="least_unsafe",
                        choices=["least_unsafe", "unfiltered"])
    parser.add_argument("--cbf_neighbour_model", type=str, default="static",
                        choices=["static", "constant_velocity"])
    parser.add_argument("--cbf_hidden", type=int, default=64)
    parser.add_argument("--cbf_layers", type=int, default=2)
    # --- Rung-1 geometry-adaptive class-K gain (proactive, alpha-only) ---------
    parser.add_argument("--cbf_adaptive", action="store_true",
                        help="ramp the class-K gain alpha per agent from --cbf_alpha "
                             "toward --cbf_alpha_max using the local conflict-geometry "
                             "(liveness) signal; horizon stays fixed at --cbf_horizon. "
                             "Rung 1 of the geometry-adaptive filter.")
    parser.add_argument("--cbf_alpha_max", type=float, default=2.0,
                        help="upper end of the adaptive alpha ramp (used only with "
                             "--cbf_adaptive).")
    parser.add_argument("--cbf_ell_thresh", type=float, default=0.32,
                        help="conflict angle (rad) at/below which the alpha ramp is "
                             "full (Chandra et al. liveness threshold ~0.32).")
    parser.add_argument("--cbf_ell_hi", type=float, default=0.80,
                        help="conflict angle (rad) above which the alpha ramp is off.")
    parser.add_argument("--cbf_sense_range", type=float, default=None,
                        help="km; only neighbours within this range feed the adaptive "
                             "signal. Default None = every other agent counts.")
    parser.add_argument("--cbf_heading_gate", type=float, default=0.6,
                        help="rad; a neighbour whose heading is within this of the "
                             "ego's is treated as in-line following and excluded from "
                             "the adaptive ramp (so corridor queueing stays at base "
                             "alpha; only crossing/opposing geometry ramps). 0 disables "
                             "the gate (ungated Rung-1 behaviour).")
    parser.add_argument("--cbf_adaptive_horizon", action="store_true",
                        help="Rung 2: also ramp the per-agent look-ahead horizon DOWN "
                             "from --cbf_horizon toward --cbf_h_min on the same "
                             "conflict-geometry signal (more convergence -> shorter "
                             "horizon). Use together with --cbf_adaptive.")
    parser.add_argument("--cbf_h_min", type=int, default=1,
                        help="shortest horizon the adaptive ramp can reach (used only "
                             "with --cbf_adaptive_horizon). Default 1.")
    parser.add_argument("--cbf_gap_guard", action="store_true",
                        help="Variant 1 junction gap guard: suppress the adaptive ramp "
                             "(force base alpha / deployed horizon) for any agent NOT "
                             "physically inside its corridor tube. Removes the 90-deg "
                             "junction turn-artifact over-ramp in the sequential bend "
                             "while in-tube merge convergence still ramps. Uses the "
                             "scenario's own is_in_tube (same test as the in/out metric).")
    parser.add_argument("--run_stage", type=str, default=None)
    parser.add_argument("--policy_slug", type=str, default=None)
    parser.add_argument("--eval_tag", type=str, default=None)
    parser.add_argument("--cbf_show", action="store_true",
                        help="draw the per-step render while evaluating (watch it "
                             "live, or save the GIF if --save_gifs). OFF by default: "
                             "a CBF eval run is metrics-only so it is fast and cannot "
                             "hit the render-buffer OOM. The numbers are identical "
                             "either way -- this only controls whether a frame is drawn.")
    parser.add_argument("--cbf_device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="device for the barrier network inside the safety filter. "
                             "cpu (default) is fully reproducible. cuda runs the barrier "
                             "forward passes on the GPU (faster for a FILTERED run, which "
                             "is barrier-bound); auto = cuda if available else cpu. NOTE: "
                             "CPU and GPU float arithmetic differ at ~1e-7, which can flip "
                             "a few borderline keep/drop decisions, so FILTERED numbers may "
                             "shift very slightly vs cpu. The policy/env stay on CPU, so the "
                             "BASELINE (no-filter) numbers are unaffected.")
    # === end CBF evaluation args ===

    all_args = parser.parse_known_args(args)[0]

    return all_args


def modify_args(model_dir:str, 
                args:argparse.Namespace, 
                exclude_args:list=['model_dir', 'model_dir_slow', 'slow_agent_ids',
                                'num_agents', 'num_obstacles', 
                                'num_landmarks', 'render_episodes', 'world_size',
                                'seed', 'save_gifs', 'use_render', 'episode_length',
                                'use_dones', 'goal_rew','collaborative', 
                                'min_dist_thresh', 'scenario_name','fair_rew','model_name',
                                'num_walls','zero_shift','min_obs_dist','max_edge_dist',
                                'total_actions', 'formation_type', 'formation_rew']):
    """
    Update args using model config.yaml — but only fill missing values.
    User-provided CLI args take priority.
    """
    import yaml
    with open(str(model_dir) + '/config.yaml') as f:
        ydict = yaml.safe_load(f)

    print('_'*50)
    for k, v in ydict.items():
        if k in exclude_args:
            print(f"Keeping CLI-provided {k} = {vars(args)[k]}")
            continue
        if isinstance(v, dict) and 'value' in v:
            # only update if CLI didn't specify
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v['value'])
    print('_'*50)

    # manual overrides
    args.cuda = False
    args.use_wandb = False
    args.use_render = True
    args.n_rollout_threads = 1
    return args


# === CBF evaluation helpers (added) ==========================================
# air_taxi constants (mirrored from multiagent/config.AirTaxiConfig), used only
# to build the world-frame safety filter.
_KTS = 0.514444 * 0.001
_CBF_V_MIN = 60 * _KTS
_CBF_V_MAX = 175 * _KTS
_CBF_ACCEL_MIN = -0.001
_CBF_ACCEL_MAX = 0.002
_CBF_ANG_RATE_MAX = 0.075
_CBF_N_ACCEL = 5
_CBF_N_ANGRATE = 5
_CBF_DT = 1.0
_CBF_SEP = 500 * 0.0003048
_CBF_MAX_EDGE = 2 * 1.60934


def _attach_cbf_eval(runner, all_args, device):
    """Attach the CBF metrics collector (and optionally the safety filter) to the
    runner so render() picks them up. Does nothing unless --cbf_eval is set."""
    if not getattr(all_args, "cbf_eval", False):
        return

    # per-seed list for the result filename (one seed per eval_mpe invocation)
    all_args._seeds = [int(all_args.seed)]
    if not getattr(all_args, "eval_tag", None):
        all_args.eval_tag = getattr(all_args, "run_stage", None) or "eval"

    from onpolicy.algorithms.utils.cbf_eval_metrics import CBFEvalCollector

    filtered = bool(getattr(all_args, "cbf_filter", False))
    if filtered:
        from onpolicy.algorithms.utils.cbf_gnn import CBFGNN
        from onpolicy.algorithms.utils.cbf_filter import (
            CBFSafetyFilter, build_primitive_table)
        # Barrier device, chosen independently of the (forced-CPU) policy/env device.
        # Only the barrier's GNN forward passes move; all of the filter's geometry
        # stays in CPU numpy, so moving the barrier to CUDA speeds up a barrier-bound
        # FILTERED run without touching the BASELINE path.
        _want = getattr(all_args, "cbf_device", "cpu")
        if _want == "auto":
            _want = "cuda" if torch.cuda.is_available() else "cpu"
        if _want == "cuda" and not torch.cuda.is_available():
            print("[cbf] --cbf_device cuda requested but no GPU is visible; using cpu")
            _want = "cpu"
        bdev = torch.device(_want)
        prim_table = build_primitive_table(_CBF_N_ACCEL, _CBF_N_ANGRATE,
                                           _CBF_ACCEL_MIN, _CBF_ACCEL_MAX,
                                           _CBF_ANG_RATE_MAX)
        barrier = CBFGNN(node_obs_dim=7, edge_dim=1,
                         hidden_size=all_args.cbf_hidden,
                         num_layers=all_args.cbf_layers,
                         max_edge_dist=_CBF_MAX_EDGE).to(bdev)
        barrier.load_state_dict(torch.load(all_args.cbf_barrier, map_location=bdev))
        barrier.eval()
        runner.cbf_eval_filter = CBFSafetyFilter(
            barrier=barrier, prim_table=prim_table, sep=_CBF_SEP, dt=_CBF_DT,
            v_min=_CBF_V_MIN, v_max=_CBF_V_MAX,
            alpha=all_args.cbf_alpha, eps=all_args.cbf_eps,
            horizon=all_args.cbf_horizon, fallback=all_args.cbf_fallback,
            num_agents=all_args.num_agents,
            neighbour_model=all_args.cbf_neighbour_model, device=bdev,
            adaptive_alpha=all_args.cbf_adaptive,
            alpha_max=all_args.cbf_alpha_max,
            ell_thresh=all_args.cbf_ell_thresh, ell_hi=all_args.cbf_ell_hi,
            sense_range=all_args.cbf_sense_range,
            heading_gate=all_args.cbf_heading_gate,
            adaptive_horizon=all_args.cbf_adaptive_horizon,
            h_min=all_args.cbf_h_min,
            gap_guard=all_args.cbf_gap_guard)
        if all_args.cbf_adaptive or all_args.cbf_adaptive_horizon:
            _hz_desc = (f"horizon {all_args.cbf_horizon}->{all_args.cbf_h_min} (ADAPTIVE)"
                        if all_args.cbf_adaptive_horizon
                        else f"horizon={all_args.cbf_horizon} (fixed)")
            print(f"[cbf] safety filter attached (neighbour={all_args.cbf_neighbour_model}, "
                  f"ADAPTIVE alpha {all_args.cbf_alpha}->{all_args.cbf_alpha_max} "
                  f"[ell_thresh={all_args.cbf_ell_thresh}, ell_hi={all_args.cbf_ell_hi}, "
                  f"sense={all_args.cbf_sense_range}, heading_gate={all_args.cbf_heading_gate}, "
                  f"gap_guard={all_args.cbf_gap_guard}], "
                  f"{_hz_desc}, "
                  f"fallback={all_args.cbf_fallback}); barrier={all_args.cbf_barrier}; "
                  f"barrier_device={bdev}")
        else:
            print(f"[cbf] safety filter attached (neighbour={all_args.cbf_neighbour_model}, "
                  f"alpha={all_args.cbf_alpha}, horizon={all_args.cbf_horizon}, "
                  f"fallback={all_args.cbf_fallback}); barrier={all_args.cbf_barrier}; "
                  f"barrier_device={bdev}")
    else:
        print("[cbf] metrics-only (no filter): recording severity for the raw policy")

    runner.cbf_eval = CBFEvalCollector(filtered=filtered)


def main(args):
    # model_dir = 'trained_models/navigation/Navigation/rmappo/wandb/offline-run-20210720_220614-1eqhk4l1/files'
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args = modify_args(all_args.model_dir, all_args)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy and all_args.use_naive_recurrent_policy) == False, (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    assert all_args.use_render, ("Need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads==1, ("only support to use 1 env to render.")
    
    device = torch.device("cpu")

    # run dir
    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    run_dir = None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            raise NotImplementedError
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
    
    print_args(config['all_args'])

    runner = Runner(config)
    # actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    # runner.policy.actor.load_state_dict(actor_state_dict)

    # === CBF evaluation (added) — attach the metrics collector and (optionally)
    # the safety filter to the runner before rendering. No-op when --cbf_eval
    # is not set, so the original path is untouched. ===
    _attach_cbf_eval(runner, all_args, device)
    # === end CBF evaluation ===

    # For a CBF eval run, default to metrics-only mode: the per-step frame (a
    # 700x700 draw every step) is skipped, but every metric is computed exactly
    # the same, so the numbers are identical to runner.render(False) while the run
    # is far faster and carries no risk of the render-buffer OOM. Pass --cbf_show
    # to draw the frame anyway (watch it live, or save the GIF with --save_gifs).
    # A plain (non-CBF) invocation still renders as the original did.
    _cbf_metrics_only = (bool(getattr(all_args, "cbf_eval", False))
                         and not getattr(all_args, "cbf_show", False))
    # When watching a CBF run, use the live window (envs.render('human')) rather
    # than accumulating GIF frames: it shows exactly the same thing, step for step,
    # but holds no frame buffer, so a long watched run can't hit the OOM that GIF
    # saving caused earlier.
    if getattr(all_args, "cbf_eval", False) and getattr(all_args, "cbf_show", False):
        all_args.save_gifs = False
    runner.render(get_metrics=_cbf_metrics_only)

    # --- adaptive-ramp diagnostics (how often / how far the ramp engaged) -----
    _filt = getattr(runner, "cbf_eval_filter", None)
    if _filt is not None and (getattr(_filt, "adaptive_alpha", False)
                              or getattr(_filt, "adaptive_horizon", False)):
        st = _filt.adaptive_stats
        n = max(1, st["calls"])
        mean_alpha = st["sum_alpha"] / n
        mean_min_ang = st["sum_min_ang"] / n
        frac_ramped = st["ramped"] / n
        min_ang = st["min_min_ang"] if st["min_min_ang"] != float("inf") else float("nan")
        _hz_part = ""
        if getattr(_filt, "adaptive_horizon", False):
            mean_hz = st["sum_horizon"] / n
            _hz_part = (f" mean_horizon={mean_hz:.2f} min_horizon={st['min_horizon']} "
                        f"(h {_filt.horizon}->{_filt.h_min})")
        print(f"[cbf-adaptive] agent-steps={st['calls']} "
              f"ramped={frac_ramped:.1%} mean_alpha={mean_alpha:.3f} "
              f"max_alpha={st['max_alpha']:.3f} "
              f"mean_min_angle={mean_min_ang:.3f}rad min_min_angle={min_ang:.3f}rad "
              f"(ell_thresh={_filt.ell_thresh}, ell_hi={_filt.ell_hi}, "
              f"heading_gate={_filt.heading_gate}){_hz_part}"
              + (f" gap_guard_overrides={st['guard'] / n:.1%}" if _filt.gap_guard else ""))

    # --- test-binding diagnostic: which test actually blocks moves? -----------
    if _filt is not None and hasattr(_filt, "bind_stats"):
        for _reg, b in _filt.bind_stats.items():
            if b["steps"] == 0:
                continue
            c, n = max(1, b["checks"]), b["steps"]
            print(f"[cbf-binding] {_reg}: agent-steps={n}  move-checks={b['checks']}  "
                  f"passed={b['passed'] / c:.1%}  barrier-only={b['barrier_only'] / c:.1%}  "
                  f"separation-only={b['separation_only'] / c:.1%}  both={b['both'] / c:.1%}  "
                  f"| steps where barrier removed a move the separation test allowed="
                  f"{b['steps_barrier_extra'] / n:.1%}  steps where barrier alone left no move="
                  f"{b['steps_barrier_causes_none'] / n:.1%}")

    # post process
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
