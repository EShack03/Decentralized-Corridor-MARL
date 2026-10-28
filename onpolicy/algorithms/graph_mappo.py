import time
import numpy as np
import argparse
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from onpolicy.utils.graph_buffer import GraphReplayBuffer
from onpolicy.utils.util import get_grad_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
import torch.jit as jit
import torch.cuda.amp as amp

# @jit.script
# def compute_policy_loss(imp_weights: torch.Tensor, adv_targ: torch.Tensor, clip_param: float) -> torch.Tensor:
#     surr1 = imp_weights * adv_targ
#     surr2 = torch.clamp(imp_weights, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
#     return -torch.min(surr1, surr2).mean()
class GR_MAPPO():
    """
        Trainer class for Graph MAPPO to update policies.
        args: (argparse.Namespace)  
            Arguments containing relevant model, policy, and env information.
        policy: (GR_MAPPO_Policy) 
            Policy to update.
        device: (torch.device) 
            Specifies the device to run on (cpu/gpu).
    """
    def __init__(self, 
                args:argparse.Namespace, 
                policy:GR_MAPPOPolicy,
                device=torch.device("cpu")) -> None:

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        # print(f'Use recurrent policy: {self._use_recurrent_policy}')
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.scaler = amp.GradScaler()

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

        # ---- Stage-3B: trainable barrier (co-adaptation). Only built when
        # --cbf_train_barrier is set; otherwise everything below is inert and
        # training is unchanged. The SAME CBFGNN instance is shared with the
        # rollout filter (wired in train_mpe.py) so the filter always uses the
        # currently-learning barrier. ----
        self.cbf_train_barrier = getattr(args, "cbf_train_barrier", False)
        self.cbf_barrier = None
        self.barrier_optimizer = None
        if self.cbf_train_barrier:
            from onpolicy.algorithms.utils.cbf_gnn import CBFGNN, safe_unsafe_masks
            self._safe_unsafe_masks = safe_unsafe_masks
            self.cbf_num_agents = args.num_agents
            self.cbf_sep = 500 * 0.0003048            # COLLISION_DISTANCE (km)
            self.cbf_max_edge = 2 * 1.60934           # COORDINATION_RANGE (km)
            self.cbf_alpha = getattr(args, "cbf_alpha", 1.0)
            self.cbf_eps = getattr(args, "cbf_eps", 0.02)
            self.cbf_margin = getattr(args, "cbf_margin", 2.0)
            self.cbf_loss_coef = getattr(args, "cbf_loss_coef", 1.0)
            self.cbf_barrier = CBFGNN(
                node_obs_dim=7, edge_dim=1,
                hidden_size=getattr(args, "cbf_hidden", 64),
                num_layers=getattr(args, "cbf_layers", 2),
                max_edge_dist=self.cbf_max_edge).to(self.device)
            init_path = getattr(args, "cbf_barrier", None)
            if init_path:
                self.cbf_barrier.load_state_dict(
                    torch.load(init_path, map_location=self.device))
                print(f"[3B] barrier initialised from {init_path} (will keep learning)")
            self.barrier_optimizer = torch.optim.Adam(
                self.cbf_barrier.parameters(),
                lr=getattr(args, "cbf_lr", 1e-3))

    def cal_value_loss(self, 
                    values:Tensor, 
                    value_preds_batch:Tensor, 
                    return_batch:Tensor, 
                    active_masks_batch:Tensor) -> Tensor:
        """
            Calculate value function loss.
            values: (torch.Tensor) 
                value function predictions.
            value_preds_batch: (torch.Tensor) 
                "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) 
                reward to go returns.
            active_masks_batch: (torch.Tensor) 
                denotes if agent is active or dead at a given timesep.

            :return value_loss: (torch.Tensor) 
                value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - 
                            value_preds_batch).clamp(-self.clip_param,
                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - \
                            value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - \
                            values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def compute_cbf_loss(self, node_obs_batch, adj_batch, agent_id_batch,
                         next_node_obs_batch, next_adj_batch, next_masks_batch):
        """
        Stage-3B barrier CBF loss on a minibatch of ego-graphs. Trains the barrier
        to keep h>=0 in safe states, h<0 in unsafe states, and satisfy the barrier
        condition h_dot + alpha*h >= 0 along real consecutive transitions (h_dot
        masked out where the transition crosses an episode reset).
        Returns a scalar loss tensor (0 if a needed batch is empty).
        """
        no = check(node_obs_batch).to(**self.tpdv)
        ad = check(adj_batch).to(**self.tpdv)
        nno = check(next_node_obs_batch).to(**self.tpdv)
        nad = check(next_adj_batch).to(**self.tpdv)
        ego = check(agent_id_batch).to(self.device).long().reshape(-1)      # (B,)
        valid = check(next_masks_batch).to(**self.tpdv).reshape(-1)         # (B,) 1=valid
        B = no.shape[0]
        na = self.cbf_num_agents
        ar = torch.arange(B, device=self.device)

        h = self.cbf_barrier(no, ad)[ar, ego]                              # (B,)
        h_next = self.cbf_barrier(nno, nad)[ar, ego]                       # (B,)

        # nearest OTHER-agent distance from the ego's row of adj (for safe/unsafe labels)
        row = ad[ar, ego][:, :na].clone()                                  # (B, na)
        row[ar, ego] = float("inf")                                        # ignore self
        row[row == 0.0] = float("inf")                                     # ignore disconnected
        nn_dist = row.min(dim=1).values                                    # (B,)

        safe_mask, unsafe_mask = self._safe_unsafe_masks(
            nn_dist, self.cbf_sep, margin=self.cbf_margin)
        eps = self.cbf_eps

        zero = h.sum() * 0.0
        loss_safe = torch.relu(-h[safe_mask] + eps).mean() if safe_mask.any() else zero
        loss_unsafe = torch.relu(h[unsafe_mask] + eps).mean() if unsafe_mask.any() else zero

        # barrier condition on valid (non-reset) transitions only
        h_dot = h_next - h
        cond = torch.relu(-(h_dot) - self.cbf_alpha * h + eps)             # (B,)
        denom = valid.sum()
        loss_hdot = (cond * valid).sum() / denom if denom > 0 else zero

        return 1.0 * loss_safe + 1.0 * loss_unsafe + 0.1 * loss_hdot
    @torch.cuda.amp.autocast()
    def ppo_update(self, 
                sample:Tuple, 
                update_actor:bool=True,
                accumulation_steps: int = 1) -> Tuple[Tensor, Tensor, 
                                                Tensor, Tensor, 
                                                Tensor, Tensor]:
        """
            Update actor and critic networks.
            sample: (Tuple) 
                contains data batch with which to update networks.
            update_actor: (bool) 
                whether to update actor network.

            :return value_loss: (torch.Tensor) 
                value function loss.
            :return critic_grad_norm: (torch.Tensor) 
                gradient norm from critic update.
            ;return policy_loss: (torch.Tensor) 
                actor(policy) loss value.
            :return dist_entropy: (torch.Tensor) 
                action entropies.
            :return actor_grad_norm: (torch.Tensor) 
                gradient norm from actor update.
            :return imp_weights: (torch.Tensor) 
                importance sampling weights.
        """
        share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
        share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, \
        actions_batch, value_preds_batch, return_batch, masks_batch, \
        active_masks_batch,old_action_log_probs_batch, adv_targ, \
        available_actions_batch = sample[:16]
        # Stage-3B: the recurrent generator appends next-step graphs (19-tuple);
        # other generators yield 16. Grab the extras if present.
        if len(sample) >= 19:
            next_node_obs_batch, next_adj_batch, next_masks_batch = sample[16:19]
        else:
            next_node_obs_batch = next_adj_batch = next_masks_batch = None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        # print("ACTIVEMaPPO", active_masks_batch.T)
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                        share_obs_batch,
                                                        obs_batch,
                                                        node_obs_batch,
                                                        adj_batch,
                                                        agent_id_batch,
                                                        share_agent_id_batch,
                                                        rnn_states_batch, 
                                                        rnn_states_critic_batch, 
                                                        actions_batch, 
                                                        masks_batch, 
                                                        available_actions_batch,
                                                        active_masks_batch)
        # actor update
        # print(f'obs: {obs_batch.shape}')
        # st = time.time()
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # print(f'Imp weights: {imp_weights.shape}')
        # print("action_log_probs",action_log_probs.shape)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 
                            1.0 + self.clip_param) * adv_targ
        # print(f'Surr1: {surr1.shape} \t Surr2: {surr2.shape} \t Values: {values.shape} \t self._use_policy_active_masks: {self._use_policy_active_masks}')
        # policy_loss = compute_policy_loss(imp_weights, adv_targ, self.clip_param)
        # print(f'Policy loss: {policy_loss.shape}', policy_loss)
        # print( -torch.sum(torch.min(surr1, surr2), dim=-1,
        #                         keepdim=True))
        # print(-torch.sum(torch.min(surr1, surr2), 
        #                                     dim=-1, keepdim=True).mean())
        # print("active masks", active_masks_batch.T)
        if self._use_policy_active_masks:
            policy_action_loss = (
                                -torch.sum(torch.min(surr1, surr2), dim=-1,
                                keepdim=True) * active_masks_batch
                                ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), 
                                            dim=-1, keepdim=True).mean()
        # print('policy_action_loss:', policy_action_loss)

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        # print(f'Actor Zero grad time: {time.time() - st}')
        st = time.time()

        if update_actor:
            self.scaler.scale((policy_loss - dist_entropy * self.entropy_coef)).backward()
            # (policy_loss - dist_entropy * self.entropy_coef).backward()
        critic_backward_time = time.time() - st
        # print(f'Actor Backward time: {critic_backward_time}')
        # st = time.time()
        self.scaler.unscale_(self.policy.actor_optimizer)
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                                            self.policy.actor.parameters(), 
                                            self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())




        # self.policy.actor_optimizer.step()

        # Accumulate gradients without stepping the optimizer
        if update_actor and (self.update_counter + 1) % accumulation_steps == 0:
            self.scaler.step(self.policy.actor_optimizer) #.step()
            # self.policy.actor_optimizer.step()
            # self.policy.actor_optimizer.zero_grad()




        # print(f'Actor Step time: {time.time() - st}')
        st = time.time()

        # critic update
        # print("values",values.shape, value_preds_batch.shape)
        value_loss = self.cal_value_loss(values, 
                                        value_preds_batch, 
                                        return_batch, 
                                        active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        # print(f'Critic Zero grad time: {time.time() - st}')
        
        st = time.time()
        critic_loss = (value_loss * self.value_loss_coef)   # TODO add gradient accumulation here
        # critic_loss.backward()
        self.scaler.scale(critic_loss).backward()
        actor_backward_time = time.time() - st
        # print(f'Critic Backward time: {actor_backward_time}')
        # st = time.time()
        self.scaler.unscale_(self.policy.critic_optimizer)
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                                                self.policy.critic.parameters(), 
                                                self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        # self.policy.critic_optimizer.step()
        # Accumulate gradients without stepping the optimizer
        if (self.update_counter + 1) % accumulation_steps == 0:
            self.scaler.step(self.policy.critic_optimizer) #.step()
            # self.policy.critic_optimizer.step()
            # self.policy.critic_optimizer.zero_grad()

        # ---- Stage-3B: barrier CBF-loss update (co-adaptation) ----
        # Separate zero_grad/backward/step for the barrier, mirroring actor/critic
        # and sharing the single AMP scaler. Only runs in 3B mode with next-step
        # graphs present. Trains ONLY the barrier's params (its own optimiser).
        cbf_loss_val = 0.0
        if self.cbf_train_barrier and (next_node_obs_batch is not None):
            cbf_loss = self.compute_cbf_loss(
                node_obs_batch, adj_batch, agent_id_batch,
                next_node_obs_batch, next_adj_batch, next_masks_batch)
            # Guard: never apply a non-finite CBF loss to the barrier — a single
            # NaN/Inf gradient would silently corrupt it for the whole run.
            if torch.isfinite(cbf_loss):
                self.barrier_optimizer.zero_grad()
                self.scaler.scale(cbf_loss * self.cbf_loss_coef).backward()
                self.scaler.unscale_(self.barrier_optimizer)
                if self._use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.cbf_barrier.parameters(),
                                             self.max_grad_norm)
                if (self.update_counter + 1) % accumulation_steps == 0:
                    self.scaler.step(self.barrier_optimizer)
                cbf_loss_val = float(cbf_loss.detach())
            else:
                print("[3B][warn] non-finite CBF loss this minibatch — "
                      "barrier update skipped")

        # Increment the update counter
        self.update_counter += 1
        self.scaler.update()

        # print(f'Critic Step time: {time.time() - st}')
        # print('_'*50)

        return (value_loss, critic_grad_norm, policy_loss,
                dist_entropy, actor_grad_norm, imp_weights,
                actor_backward_time, critic_backward_time, cbf_loss_val)

    def train(self, 
            buffer:GraphReplayBuffer, 
            update_actor:bool=True, accumulation_steps: int = 1):
        """
            Perform a training update using minibatch GD.
            buffer: (GraphReplayBuffer) 
                buffer containing training data.
            update_actor: (bool) 
                whether to update actor network.

            :return train_info: (dict) 
                contains information regarding 
                training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - \
                        self.value_normalizer.denormalize(
                                                buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        # print("TRAINBUFFER", buffer.active_masks[:-1].T)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cbf_loss'] = 0   # Stage-3B barrier loss

        self.update_counter = 0

        for _ in range(self.ppo_epoch):
            st = time.time()
            if self._use_recurrent_policy:
                # print("Recurrent")
                data_generator = buffer.recurrent_generator(advantages, 
                                                        self.num_mini_batch, 
                                                        self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, 
                                                            self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, 
                                                            self.num_mini_batch)
            
            # actor_backward_time, critic_backward_time = 0, 0 

            for sample in data_generator:
                # print("Sample", sample)

                value_loss, critic_grad_norm, policy_loss, dist_entropy, \
                actor_grad_norm, imp_weights, actor_bt, critic_bt, cbf_loss_val = self.ppo_update(sample, update_actor, accumulation_steps)


                # actor_backward_time += actor_bt
                # critic_backward_time += critic_bt
                train_info['cbf_loss'] += cbf_loss_val
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

            # print(f'PPO epoch time: {time.time() - st}')
            # print(f'PPO epoch actor backward time: {actor_backward_time}')
            # print(f'PPO epoch critic backward time: {critic_backward_time}')

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Convert networks to training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Convert networks to eval mode"""
        self.policy.actor.eval()
        self.policy.critic.eval()
