#!/usr/bin/env python
import argparse
from distutils.util import strtobool 
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

import os,sys
sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box, connected_to_internet
from onpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, GraphSubprocVecEnv, GraphDummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args:argparse.Namespace):
    def get_env_fn(rank:int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
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
        return SubprocVecEnv([get_env_fn(i) for i in 
                                range(all_args.n_rollout_threads)])


def make_eval_env(all_args:argparse.Namespace):
    def get_env_fn(rank:int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv([get_env_fn(i) for i in 
                                        range(all_args.n_rollout_threads)])
        return SubprocVecEnv([get_env_fn(i) for i in 
                                range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='navigation_graph', help="Which scenario to run on")
    parser.add_argument('--num_landmarks', type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=2, 
                        help="number of players")
    parser.add_argument('--num_obstacles', type=int, default=0, 
                        help="Number of obstacles")
    parser.add_argument("--collaborative", type=lambda x:bool(strtobool(x)), 
                    default=False, help="Number of agents in the env")
    parser.add_argument("--max_speed", type=float, default=2, 
                        help="Max speed for agents. NOTE that if this is None, "
                        "then max_speed is 2 with discrete action space")

    
    #parser.add_argument("--fair_wt", type=int,
    #                    default=1,help="choice of weight")
    #parser.add_argument("--fair_rew", type=int,
    #                    default=1, help ="choice of fairness reward")
    parser.add_argument("--min_dist_thresh", type=float, default=0.05, 
                        help="The minimum distance threshold to classify whether "
                        "agent has reached the goal or not")

    parser.add_argument("--use_dones", type=lambda x:bool(strtobool(x)), 
                        default=False, help="Whether we want to use the 'done=True' " 
                        "when agent has reached the goal or just return False like "
                        "the `simple.py` or `simple_spread.py`")

    '''
    parser.add_argument("--dynamics_type", type=str, default="air_taxi",
    choices=["unicycle_vehicle", "double_integrator", "air_taxi"],
    help="Type of vehicle dynamics to use.")
    '''

    # ---- Stages 3A/3B: CBF barrier in the training loop (all off by default, so
    # omitting these leaves training byte-for-byte identical to the baseline) ----
    parser.add_argument("--cbf_train_barrier", action="store_true", default=False,
                        help="Load the CBF barrier into the trainer so the in-loop "
                        "filter and the reward shaping can use it (name kept for "
                        "compatibility with the 3A/3B training commands; the barrier "
                        "itself is not optimised).")
    parser.add_argument("--cbf_barrier", type=str, default=None,
                        help="Path to the Stage-1 barrier .pt to load. If omitted, "
                        "the barrier is randomly initialised.")
    parser.add_argument("--cbf_reward_coef", type=float, default=0.0,
                        help="lambda for barrier reward shaping (Part B). 0 = off.")
    parser.add_argument("--cbf_reward_margin", type=float, default=2.0,
                        help="Shaping margin: penalise while h_ego < this value.")
    parser.add_argument("--cbf_alpha", type=float, default=1.0,
                        help="alpha in the barrier condition h_dot + alpha*h >= 0.")
    parser.add_argument("--cbf_hidden", type=int, default=64,
                        help="Hidden size of the barrier GNN.")
    parser.add_argument("--cbf_layers", type=int, default=2,
                        help="Number of layers in the barrier GNN.")
    parser.add_argument("--cbf_filter_in_loop",
                        type=lambda x: bool(strtobool(x)), default=True,
                        help="Run the ego-frame CBF filter in the training "
                        "rollout (shielded RL, as in 3A). Set False to train "
                        "the policy with NO filter mask (pure reward-shaping "
                        "ablation). Only has effect when --cbf_train_barrier.")
    parser.add_argument("--cbf_filter_eps", type=float, default=0.0,
                        help="Slack eps in the filter's barrier-condition test.")
    parser.add_argument("--cbf_neighbour_model", type=str, default="static",
                        choices=["static", "constant_velocity"],
                        help="Neighbour motion the filter assumes over its "
                        "one-step lookahead. static keeps Stages 2/3A/3B "
                        "consistent; constant_velocity is the committed upgrade.")

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser


def main(args):
    parser = get_config()
    all_args, parser = parse_args(args, parser)
    if all_args.env_name == "GraphMPE":
        from onpolicy.config import graph_config
        all_args, parser = graph_config(args, parser)

    if all_args.algorithm_name in ["rmappo"]:
        assert (all_args.use_recurrent_policy or 
                all_args.use_naive_recurrent_policy), (
                "check recurrent policy!")
    elif all_args.algorithm_name in ["mappo"]:
        assert (all_args.use_recurrent_policy == False and 
                all_args.use_naive_recurrent_policy == False), (
                "check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and 
            all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. "
        "Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    if all_args.verbose:
        print_args(all_args)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + 
        "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir),exist_ok=True)

    # wandb
    if all_args.use_wandb:

        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json
            # save a json file with your wandb api key in your 
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser('~')+'/keys.json') as json_file: 
                key = json.load(json_file)
                my_wandb_api_key = key['my_wandb_api_key'] # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ['WANDB_SAVE_CODE'] = "true"

        print_box('Creating wandboard...')
        run = wandb.init(config=all_args,
                        project=all_args.project_name,
                        # project=all_args.env_name,
                        entity=all_args.user_name,
                        notes=socket.gethostname(),
                        name= str(all_args.experiment_name) +
                        "_seed" + str(all_args.seed),
                        # group=all_args.scenario_name,
                        dir=str(run_dir),
                        save_code=True,
                        # job_type="training",
                        reinit=True)
        wandb.run.log_code(".")  # save code
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for 
                            folder in run_dir.iterdir() if 
                            str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                            str(all_args.env_name) + "-" + \
                            str(all_args.experiment_name) + "@" + \
                            str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    if all_args.use_wandb:
        config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'multiagent', 'config.py'))
        wandb.save(config_file_path, policy='now')

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

    runner = Runner(config)

    # ---- Stage-3B: attach the ego-frame CBF safety filter to the runner ----
    # Built AFTER the runner so it shares the trainer's cbf_barrier instance:
    # the filter and the reward shaping then use one and the same barrier.
    if (getattr(all_args, "cbf_train_barrier", False) and
            getattr(all_args, "cbf_filter_in_loop", True)):
        from onpolicy.algorithms.utils.cbf_filter import (
            EgoCBFFilter, build_primitive_table)
        from multiagent.config import AirTaxiConfig as _AT
        prim_table = build_primitive_table(
            _AT.MOTION_PRIM_ACCEL_OPTIONS, _AT.MOTION_PRIM_ANGRATE_OPTIONS,
            _AT.ACCEL_MIN, _AT.ACCEL_MAX, _AT.ANGULAR_RATE_MAX)
        runner.cbf_filter = EgoCBFFilter(
            barrier=runner.trainer.cbf_barrier,
            prim_table=prim_table,
            sep=_AT.COLLISION_DISTANCE,
            dt=_AT.DT,
            v_min=_AT.V_MIN, v_max=_AT.V_MAX,
            alpha=getattr(all_args, "cbf_alpha", 1.0),
            eps=getattr(all_args, "cbf_filter_eps", 0.0),
            num_agents=all_args.num_agents,
            neighbour_model=getattr(all_args, "cbf_neighbour_model", "static"),
            fallback="least_unsafe",
            device=device)
        print(f"[3B] ego-frame CBF filter attached "
              f"(neighbour_model={getattr(all_args, 'cbf_neighbour_model', 'static')}, "
              f"alpha={getattr(all_args, 'cbf_alpha', 1.0)}, "
              f"eps={getattr(all_args, 'cbf_filter_eps', 0.0)})")

    if all_args.verbose:
        print_box('Actor Network', 80)
        if type(runner.policy)==list:
            print_box(runner.policy[0].actor, 80)
            print_box('Critic Network', 80)
            print_box(runner.policy[0].critic, 80)
        else:
            print_box(runner.policy.actor, 80)
            print_box('Critic Network', 80)
            print_box(runner.policy.critic, 80)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
