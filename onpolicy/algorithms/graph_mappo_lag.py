# ============================================================================
# graph_mappo_lag.py - constrained fine-tuning of the corridor policy
# ----------------------------------------------------------------------------
# [ES] added (constrained training, Nov 2026). New file: a subclass of the
# original GR_MAPPO trainer. GR_MAPPO.ppo_update (the PPO actor/critic step)
# is reused unchanged; only train() is replaced, to change the advantage the
# actor is updated with and to add the extra pieces below.
#
# Two modes (--lag_mode):
#   lagrangian  MAPPO-Lagrangian with PID-controlled multipliers (Stooke et al.
#               2020). One cost critic per cost (separation, corridor-keeping).
#               Actor advantage  A = (A_r - sum_k lam_k A_ck) / (1 + sum_k lam_k)
#               with every advantage normalised over active agent-steps.
#               lam_k = clip(Kp e + Ki I + Kd max(0, e - e_prev), 0, lam_max),
#               e = J_k / d_k - 1 (normalised error), I = max(0, I + e),
#               J_k = mean cost per aircraft per episode (EMA-smoothed),
#               d_k = budget (auto: factor x J_k of the starting policy).
#   penalty     standard MAPPO: the runner subtracts fixed weights x costs from
#               the reward before it enters the buffer; no cost critics, no
#               multipliers. Everything else below is identical, so the two
#               modes differ only in how the costs enter training.
#
# Shared by both modes:
#   warm-up     the first --lag_warmup updates do not move the actor: the reward
#               critic and the cost critics settle, and (lagrangian, auto) the
#               budgets are measured from the starting policy.
#   fallback    agent-steps where the training filter found no safe primitive
#               (the least-unsafe fallback chose the move, not the policy) get
#               advantage 0, so the policy is neither credited nor blamed.
#   KL anchor   + c_kl * KL(pi_ref || pi) on sampled agent-steps, pi_ref = the
#               frozen starting actor, so fine-tuning cannot drift far.
#   mask term   + c_mask * -log(1 - p_blocked), p_blocked = probability the
#               UNMASKED policy puts on moves the 1-second filter would block,
#               so the policy learns to avoid them itself.
# The rollout masks are already stored in the buffer (available_actions), so
# the PPO ratio uses the masked distribution the actions were sampled from
# (Huang & Ontanon 2022).
# ============================================================================
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from onpolicy.algorithms.graph_mappo import GR_MAPPO
from onpolicy.algorithms.graph_actor_critic import GR_Critic
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import huber_loss
from onpolicy.utils.valuenorm import ValueNorm

COST_NAMES = ("sep", "corr")


def _cast(x):
    """(T, threads, agents, ...) -> (threads*agents*T, ...), the buffer's order."""
    return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])


def _norm_active(a, active):
    """Normalise a over rows where active == 1 (a and active same shape)."""
    v = a[active > 0]
    if v.size == 0:
        return a * 0.0
    return (a - v.mean()) / (v.std() + 1e-5)


class GR_MAPPO_Lag(GR_MAPPO):
    def __init__(self, args, policy, device=torch.device("cpu")):
        super().__init__(args, policy, device=device)
        # GR_MAPPO builds amp.GradScaler() unconditionally. It is enabled whenever
        # the machine has CUDA, even when training runs on the CPU (--cuda off),
        # and then scale() rejects CPU tensors. Enable it only for a CUDA device.
        self.scaler = torch.cuda.amp.GradScaler(enabled=(torch.device(device).type == "cuda"))
        self.mode = args.lag_mode
        self.K = len(COST_NAMES)
        self.gamma, self.gae_lambda = args.gamma, args.gae_lambda
        self.warmup = int(args.lag_warmup)
        self.iteration = 0
        self.store = None                       # set by the runner (cost arrays)
        self.log_path = None                    # set by the runner

        # ---- cost critics (lagrangian mode only): non-recurrent copies of the
        # reward critic's architecture, each with its own ValueNorm ----
        self.cost_critics, self.cost_opts, self.cost_norms = [], [], []
        if self.mode == "lagrangian":
            cargs = copy.copy(args)
            cargs.use_recurrent_policy = False
            cargs.use_naive_recurrent_policy = False
            cargs.use_popart = False
            for _ in range(self.K):
                c = GR_Critic(cargs, policy.share_obs_space, policy.node_obs_space,
                              policy.edge_obs_space, device, policy.split_batch,
                              policy.max_batch_size)
                self.cost_critics.append(c)
                self.cost_opts.append(torch.optim.Adam(
                    c.parameters(), lr=args.critic_lr, eps=args.opti_eps,
                    weight_decay=args.weight_decay))
                self.cost_norms.append(ValueNorm(1, device=device))

        # ---- PID multipliers ----
        self.kp, self.ki, self.kd = args.lag_kp, args.lag_ki, args.lag_kd
        self.lam_max = args.lag_lambda_max
        self.ema = args.lag_cost_ema
        self.lam = np.zeros(self.K)
        self.I = np.zeros(self.K)
        self.e_prev = np.zeros(self.K)
        self.J_ema = None
        self.budget = np.array([args.lag_budget_sep, args.lag_budget_corr], float)
        self.budget_factor = np.array([args.lag_budget_factor_sep,
                                       args.lag_budget_factor_corr], float)
        self.budget_floor = args.lag_budget_floor
        self._auto = self.budget <= 0           # <= 0 means: measure it in warm-up
        self._J_warm = []
        if self.mode == "lagrangian" and self._auto.any() and self.warmup < 1:
            raise ValueError("automatic budgets need --lag_warmup >= 1")

        # ---- auxiliary actor losses ----
        self.c_kl = args.lag_kl_coef
        self.c_mask = args.lag_mask_coef
        self.aux_rows = args.lag_aux_rows
        self.zero_fallback = args.lag_zero_fallback
        self.ref_actor = None
        if self.c_kl > 0:
            self.ref_actor = copy.deepcopy(policy.actor).eval()
            for p in self.ref_actor.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------ persistence
    def lag_state(self):
        return dict(lam=self.lam, I=self.I, e_prev=self.e_prev, J_ema=self.J_ema,
                    budget=self.budget, iteration=self.iteration,
                    cost_critics=[c.state_dict() for c in self.cost_critics],
                    cost_norms=[n.state_dict() for n in self.cost_norms])

    # ------------------------------------------------------------ cost pieces
    @torch.no_grad()
    def _cost_values(self, buffer, k, chunk=4096):
        """Denormalised cost value for every (T+1, threads, agents) row."""
        c = self.cost_critics[k]
        c.eval()
        T1, E, A = buffer.masks.shape[:3]
        so = buffer.share_obs.reshape(T1 * E * A, *buffer.share_obs.shape[3:])
        no = buffer.node_obs.reshape(T1 * E * A, *buffer.node_obs.shape[3:])
        ad = buffer.adj.reshape(T1 * E * A, *buffer.adj.shape[3:])
        sid = buffer.share_agent_id.reshape(T1 * E * A, *buffer.share_agent_id.shape[3:])
        rnn = np.zeros((T1 * E * A, buffer.recurrent_N, buffer.hidden_size), np.float32)
        m = buffer.masks.reshape(T1 * E * A, 1)
        out = []
        for i in range(0, T1 * E * A, chunk):
            v, _ = c(so[i:i + chunk], no[i:i + chunk], ad[i:i + chunk],
                     sid[i:i + chunk], rnn[i:i + chunk], m[i:i + chunk])
            out.append(self.cost_norms[k].denormalize(v))
        return np.concatenate(out).reshape(T1, E, A, 1)

    def _cost_gae(self, buffer, costs, values):
        """GAE on one cost: costs (T, E, A, 1), values (T+1, E, A, 1)."""
        T = costs.shape[0]
        adv = np.zeros_like(costs)
        gae = 0.0
        for t in reversed(range(T)):
            m = buffer.masks[t + 1]
            delta = costs[t] + self.gamma * values[t + 1] * m - values[t]
            gae = delta + self.gamma * self.gae_lambda * m * gae
            adv[t] = gae
        return adv, adv + values[:-1]

    def _update_cost_critic(self, flat, k, returns, n_batches):
        """One pass of minibatch regression of cost critic k on its returns.
        flat = the buffer arrays already flattened with _cast (built once)."""
        c, opt, norm = self.cost_critics[k], self.cost_opts[k], self.cost_norms[k]
        c.train()
        so, no, ad, sid, msk, act = flat
        ret = _cast(returns)
        n = ret.shape[0]
        rnn = np.zeros((1, self.policy.critic._recurrent_N, self.policy.critic.hidden_size),
                       np.float32)
        perm = np.random.permutation(n)
        losses = []
        for idx in np.array_split(perm, max(1, n_batches)):
            r = check(ret[idx]).to(**self.tpdv)
            norm.update(r)
            v, _ = c(so[idx], no[idx], ad[idx], sid[idx],
                     np.repeat(rnn, len(idx), axis=0), msk[idx])
            err = norm.normalize(r) - v
            a = check(act[idx]).to(**self.tpdv)
            loss = (huber_loss(err, self.huber_delta) * a).sum() / a.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(c.parameters(), self.max_grad_norm)
            opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def _pid(self, J):
        """One PID step on the normalised error; returns the new multipliers."""
        e = J / np.maximum(self.budget, self.budget_floor) - 1.0
        self.I = np.maximum(0.0, self.I + e)
        d = np.maximum(0.0, e - self.e_prev)
        self.e_prev = e
        self.lam = np.clip(self.kp * e + self.ki * self.I + self.kd * d, 0.0, self.lam_max)
        return e

    # ------------------------------------------------------------ aux losses
    def _dist(self, actor, obs, node_obs, adj, agent_id, rnn, masks):
        """UNMASKED action distribution of one actor on independent rows
        (the GR_Actor.forward path up to its action head)."""
        t = actor.tpdv
        obs = check(obs).to(**t); node_obs = check(node_obs).to(**t)
        adj = check(adj).to(**t); agent_id = check(agent_id).to(**t).long()
        rnn = check(rnn).to(**t); masks = check(masks).to(**t)
        f = actor.gnn_base(node_obs, adj, agent_id)
        f = actor.base(torch.cat([obs, f], dim=1))
        if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
            f, _ = actor.rnn(f, rnn, masks)
        return actor.act.action_out(f, None)

    def _aux_step(self, buffer, safe1):
        """One actor step on the KL anchor and the mask-consistency term."""
        act = _cast(buffer.active_masks[:-1]).reshape(-1)
        rows = np.flatnonzero(act > 0)
        if rows.size == 0:
            return 0.0, 0.0
        idx = np.random.choice(rows, size=min(self.aux_rows, rows.size), replace=False)
        T1, E, A = buffer.masks.shape[:3]
        # row order of _cast: thread, agent, time
        e_i, rem = np.divmod(idx, A * (T1 - 1))
        a_i, t_i = np.divmod(rem, T1 - 1)
        sl = (t_i, e_i, a_i)
        args = (buffer.obs[sl], buffer.node_obs[sl], buffer.adj[sl],
                buffer.agent_id[sl], buffer.rnn_states[sl], buffer.masks[sl])
        self.policy.actor.train()
        dist = self._dist(self.policy.actor, *args)
        logp = torch.log_softmax(dist.logits, dim=-1)
        loss = torch.zeros((), **self.tpdv)
        kl_v = mask_v = 0.0
        if self.ref_actor is not None and self.c_kl > 0:
            with torch.no_grad():
                ref_logp = torch.log_softmax(self._dist(self.ref_actor, *args).logits, -1)
            kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean()
            loss = loss + self.c_kl * kl
            kl_v = kl.item()
        if safe1 is not None and self.c_mask > 0:
            blocked = check(1.0 - safe1[sl]).to(**self.tpdv)
            p_blk = (logp.exp() * blocked).sum(-1).clamp(max=1 - 1e-6)
            mt = -torch.log1p(-p_blk).mean()
            loss = loss + self.c_mask * mt
            mask_v = mt.item()
        self.policy.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()
        return kl_v, mask_v

    # ------------------------------------------------------------ main update
    def train(self, buffer, update_actor=True, accumulation_steps=1):
        st = self.store
        active = buffer.active_masks[:-1]
        warm = self.iteration < self.warmup
        move_actor = update_actor and not warm
        info = {}

        # reward advantage, as in GR_MAPPO.train
        if self._use_popart or self._use_valuenorm:
            adv_r = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1])
        else:
            adv_r = buffer.returns[:-1] - buffer.value_preds[:-1]
        adv = _norm_active(adv_r, active)

        # per-aircraft episode cost (one rollout = one episode)
        J = st["cost"].sum(axis=0).mean(axis=(0, 1))          # (K,)
        self.J_ema = J if self.J_ema is None else self.ema * self.J_ema + (1 - self.ema) * J

        cost_returns = []
        if self.mode == "lagrangian":
            adv_c = []
            for k in range(self.K):
                vals = self._cost_values(buffer, k)
                a_k, r_k = self._cost_gae(buffer, st["cost"][..., k:k + 1], vals)
                adv_c.append(_norm_active(a_k, active))
                cost_returns.append(r_k)
            if warm:
                self._J_warm.append(J)
                if self._auto.any():               # budget from the starting policy
                    self.budget = np.where(self._auto, np.maximum(
                        self.budget_factor * np.mean(self._J_warm, axis=0),
                        self.budget_floor), self.budget)
                e = np.zeros(self.K)
            else:
                e = self._pid(self.J_ema)
            info.update({f"lag_err_{n}": e[k] for k, n in enumerate(COST_NAMES)})
            lam = self.lam if not warm else np.zeros(self.K)
            adv = (adv - sum(lam[k] * adv_c[k] for k in range(self.K))) / (1.0 + lam.sum())

        nosafe = st.get("nosafe")
        if self.zero_fallback and nosafe is not None:
            adv = adv * (1.0 - nosafe[..., None].astype(np.float32))

        # PPO epochs (GR_MAPPO.ppo_update unchanged) + aux and cost-critic steps
        for key in ("value_loss", "policy_loss", "dist_entropy", "actor_grad_norm",
                    "critic_grad_norm", "ratio"):
            info[key] = 0.0
        kl_s = mask_s = cl_s = 0.0
        flat, n_cb = None, 1
        if cost_returns:
            flat = tuple(_cast(x) for x in (buffer.share_obs[:-1], buffer.node_obs[:-1],
                                            buffer.adj[:-1], buffer.share_agent_id[:-1],
                                            buffer.masks[:-1], buffer.active_masks[:-1]))
            n_cb = max(self.num_mini_batch, int(np.ceil(flat[0].shape[0] / 16384)))
        self.update_counter = 0
        # ppo_update calls scaler.unscale_(actor optimizer) even when the actor is
        # not updated (warm-up); before any scaler.scale() call the scale is not
        # yet created and unscale_ fails. Scaling a dummy tensor creates it.
        if self.scaler.is_enabled():
            self.scaler.scale(torch.zeros((), device=self.device))
        t0 = time.time()
        for _ in range(self.ppo_epoch):
            gen = buffer.recurrent_generator(adv, self.num_mini_batch, self.data_chunk_length)
            for sample in gen:
                vl, cgn, pl, ent, agn, imp, _, _ = self.ppo_update(
                    sample, move_actor, accumulation_steps)
                info["value_loss"] += vl.item(); info["policy_loss"] += pl.item()
                info["dist_entropy"] += ent.item(); info["actor_grad_norm"] += float(agn)
                info["critic_grad_norm"] += float(cgn); info["ratio"] += float(imp.mean())
            if move_actor and (self.c_kl > 0 or self.c_mask > 0):
                kv, mv = self._aux_step(buffer, st.get("safe1"))
                kl_s += kv; mask_s += mv
            for k in range(len(cost_returns)):
                cl_s += self._update_cost_critic(flat, k, cost_returns[k], n_cb)
        n_upd = self.ppo_epoch * self.num_mini_batch
        for key in ("value_loss", "policy_loss", "dist_entropy", "actor_grad_norm",
                    "critic_grad_norm", "ratio"):
            info[key] /= n_upd

        fb = float(nosafe[active[..., 0] > 0].mean()) if nosafe is not None and \
            (active > 0).any() else 0.0
        info.update({"lag_kl": kl_s / self.ppo_epoch, "lag_mask_term": mask_s / self.ppo_epoch,
                     "lag_cost_critic_loss": cl_s / max(1, self.ppo_epoch * len(cost_returns)),
                     "lag_fallback_rate": fb, "lag_warmup": float(warm),
                     "lag_update_s": time.time() - t0,
                     "lag_rollout_s": float(getattr(self, "rollout_s", 0.0))})
        for k, n in enumerate(COST_NAMES):
            info[f"lag_J_{n}"] = float(J[k])
            info[f"lag_budget_{n}"] = float(self.budget[k])
            info[f"lag_lambda_{n}"] = float(self.lam[k]) if not warm else 0.0
        self._log(info)
        self.iteration += 1
        return info

    def _log(self, info):
        tag = "warm-up" if info["lag_warmup"] else "train"
        print(f"[lag {self.mode} it {self.iteration} {tag}] "
              f"J sep={info['lag_J_sep']:.3f} (budget {info['lag_budget_sep']:.3f}, "
              f"lam {info['lag_lambda_sep']:.2f})  "
              f"J corr={info['lag_J_corr']:.2f} (budget {info['lag_budget_corr']:.2f}, "
              f"lam {info['lag_lambda_corr']:.2f})  fallback={info['lag_fallback_rate']:.3f} "
              f"kl={info['lag_kl']:.4f} mask={info['lag_mask_term']:.4f} "
              f"rollout {info['lag_rollout_s']:.0f}s update {info['lag_update_s']:.0f}s",
              flush=True)
        if self.log_path is None:
            return
        new = not os.path.exists(self.log_path)
        keys = ["iteration"] + sorted(k for k in info if k.startswith("lag_")) + \
            ["value_loss", "policy_loss", "dist_entropy"]
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(keys)
            w.writerow([self.iteration] + [f"{float(info[k]):.6g}" for k in keys[1:]])
