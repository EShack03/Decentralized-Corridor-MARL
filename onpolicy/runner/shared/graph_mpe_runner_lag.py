# ============================================================================
# graph_mpe_runner_lag.py - training runner for constrained fine-tuning
# ----------------------------------------------------------------------------
# [ES] added (constrained training, Nov 2026). New file: a subclass of the
# original GMPERunner. The rollout loop (GMPERunner.run) is reused unchanged;
# this class only hooks three of its methods:
#
#   insert()            reads the per-agent costs and the next-step filter
#                       masks that CostMaskEnv put in the infos, stores the
#                       costs for this step, and (penalty mode) subtracts the
#                       fixed cost penalty from the reward before it is stored
#   collect_with_mask() records, for this decision step, whether the training
#                       filter fell back (no safe primitive) and the 1-second
#                       mask, then runs the original method
#   train() / save()    hands the cost arrays to GR_MAPPO_Lag, and also saves
#                       the cost critics and multiplier state
#
# The training filter is applied through the original hook: self.cbf_filter is
# set to a small adapter whose masks() returns the masks computed in the
# workers, so collect_with_mask applies them exactly as it applied the Stage-3B
# filter (finished aircraft keep their stop-only mask). With --lag_filter False
# the adapter is not attached and the policy acts unfiltered.
#
# Two fixes to the original buffer use (both in this subclass, see insert() and
# train()): the stored rollout mask is aligned with its step, and index 0 of
# each rollout holds the current reset observation instead of the first one.
#
# Step 0 of every rollout is a fresh reset (rollout length = episode length,
# env.world_length = episode_length) and uses GMPERunner.collect, which applies
# no mask; at reset every aircraft is in its start queue.
# ============================================================================
import os
import time

import numpy as np
import torch

from onpolicy.runner.shared.graph_mpe_runner import GMPERunner
from onpolicy.algorithms.graph_mappo_lag import GR_MAPPO_Lag, COST_NAMES


class _WorkerMasks:
    """Stands in for the old in-process filter: returns the masks the workers
    computed for the current state (see CostMaskEnv)."""

    def __init__(self, runner):
        self.runner = runner

    def masks(self, node_obs, adj, obs, agent_id):
        r = self.runner
        m = r._pend_mask.reshape(-1, r._n_act)
        ns = r._pend_nosafe.reshape(-1)
        return m, ns


class GMPERunnerLag(GMPERunner):
    def __init__(self, config):
        super().__init__(config)
        a = self.all_args
        self._n_act = self.envs.action_space[0].n
        self._check_critic_restore()
        # swap in the constrained trainer (same policy object, restored weights)
        self.trainer = GR_MAPPO_Lag(a, self.policy, device=self.device)
        self.trainer.log_path = os.path.join(str(self.run_dir), "lag_log.csv")
        T, E, N = self.episode_length, self.n_rollout_threads, self.num_agents
        self.store = dict(cost=np.zeros((T, E, N, len(COST_NAMES)), np.float32))
        self.use_filter = bool(a.lag_filter)
        self.warm_filter = (not self.use_filter) and a.lag_warmup > 0
        self.use_safe1 = a.lag_mask_coef > 0
        if self.use_filter or self.warm_filter:
            self.store["nosafe"] = np.zeros((T, E, N), bool)
        if self.use_safe1:
            self.store["safe1"] = np.ones((T, E, N, self._n_act), np.float32)
        self.trainer.store = self.store
        self._pend_mask = np.ones((E, N, self._n_act), np.float32)
        self._pend_nosafe = np.zeros((E, N), bool)
        self._pend_safe1 = np.ones((E, N, self._n_act), np.float32)
        self._adapter = _WorkerMasks(self)
        self.cbf_filter = self._adapter if (self.use_filter or self.warm_filter) else None
        self.penalty = np.array([a.lag_penalty_sep, a.lag_penalty_corr], np.float32) \
            if a.lag_mode == "penalty" else None
        print(f"[lag] mode={a.lag_mode} filter={'on, horizon %d s' % a.lag_filter_horizon if self.use_filter else 'off'} "
              f"kl={a.lag_kl_coef} mask={a.lag_mask_coef} warm-up={a.lag_warmup} "
              f"scenarios={a.lag_scenarios or a.scenario_name}")

    def _check_critic_restore(self):
        """The original restore() loads critic.pt with strict=False, which would
        silently leave a mismatched critic (e.g. a missing PopArt head) at its
        random initialisation. Fine-tuning must start from the trained critic,
        so check the keys explicitly."""
        if self.model_dir is None:
            return
        sd = torch.load(str(self.model_dir) + "/critic.pt", map_location="cpu")
        res = self.policy.critic.load_state_dict(sd, strict=False)
        # PopArt's running statistics are created with nn.Parameter(...).to(device);
        # on a GPU that returns an unregistered tensor, so the original GPU run never
        # saved them (critic.pt has v_out.stddev/weight/bias only). They restart at
        # zero here; the warm-up updates (actor fixed) re-fit the critic's scale.
        popart_stats = {"v_out.mean", "v_out.mean_sq", "v_out.debiasing_term"}
        missing = [k for k in res.missing_keys if k not in popart_stats]
        if set(res.missing_keys) & popart_stats:
            print("[lag] critic.pt has no PopArt running statistics (not saved by the "
                  "original GPU run); they restart at zero and warm-up re-fits them")
        if missing or res.unexpected_keys:
            raise RuntimeError(
                "critic.pt does not match the critic built from these flags "
                f"(missing {missing}, unexpected {res.unexpected_keys}). "
                "The original model was trained with --use_valuenorm --use_popart "
                "(PopArt head on, ValueNorm off); pass the same flags.")

    def run(self):
        self._t_last = time.time()
        return super().run()

    # ------------------------------------------------------------ hooks
    def _filter_now(self):
        return self.use_filter or (self.warm_filter and
                                   self.trainer.iteration < self.trainer.warmup)

    def collect_with_mask(self, step, active_agents, active_masks, finished):
        on = self._filter_now()
        # filter off in training: the original hook sees a filter only during
        # warm-up (budget measurement), none afterwards
        self.cbf_filter = self._adapter if on else None
        if "nosafe" in self.store:
            self.store["nosafe"][step] = self._pend_nosafe if on else False
        if self.use_safe1:
            self.store["safe1"][step] = self._pend_safe1
        return super().collect_with_mask(step, active_agents, active_masks, finished)

    def insert(self, data, active_agents=None, active_masks=None, finished=None):
        infos = data[7]
        t = self.buffer.step
        E, N = self.n_rollout_threads, self.num_agents
        cost = np.zeros((E, N, len(COST_NAMES)), np.float32)
        for e in range(E):
            for i in range(N):
                inf = infos[e][i]
                cost[e, i] = inf["lag_cost"]
                if "lag_mask" in inf:
                    self._pend_mask[e, i] = inf["lag_mask"]
                    self._pend_nosafe[e, i] = inf["lag_nosafe"]
                if self.use_safe1:
                    self._pend_safe1[e, i] = inf["lag_safe1"]
        self.store["cost"][t] = cost
        if self.penalty is not None:
            rewards = data[5] - (cost * self.penalty).sum(-1, keepdims=True)
            data = data[:5] + (rewards,) + data[6:]
        out = super().insert(data, active_agents=active_agents,
                             active_masks=active_masks, finished=finished)
        # Fix 1 (alignment): GraphReplayBuffer.insert stores available_actions at
        # [step + 1] (the usual MAPPO convention: the env returns the NEXT state's
        # availability), but GMPERunner passes the mask the action at THIS step was
        # sampled under. Left as is, the PPO ratio at step t would use step t-1's
        # mask, and a move blocked at t-1 but taken at t would get ratio ~0. Store
        # it at [step] as well, so available_actions[t] is the mask used at t.
        self.buffer.available_actions[t] = data[13]
        return out

    def train(self, active_agents=None):
        now = time.time()
        self.trainer.rollout_s = now - getattr(self, "_t_last", now)
        info = super().train()
        self._t_last = time.time()
        # Fix 2 (first step): GMPERunner never calls buffer.after_update() (the
        # call is commented out in base_runner), so index 0 of every rollout keeps
        # the observation from the very first reset, and step 0 acts on it while
        # the env is in a new episode. Copy the last (post-reset) entries to 0.
        self.buffer.after_update()
        # step 0 of the next rollout is not filtered: reset its records
        if "nosafe" in self.store:
            self.store["nosafe"][0] = False
        if self.use_safe1:
            self.store["safe1"][0] = 1.0
        return info

    def save(self):
        super().save()
        torch.save(self.trainer.lag_state(), str(self.save_dir) + "/lag_state.pt")
