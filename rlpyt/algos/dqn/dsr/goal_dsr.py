from collections import namedtuple

import torch
import torch.nn as nn

from rlpyt.algos.dqn.dsr.dsr import DSR
from rlpyt.utils.misc import param_norm_

OptInfo = namedtuple("OptInfo", ["goalLoss", "goalGradNorm",
                                 "dsrLoss", "dsrGradNorm", "tdAbsErr"])


class GoalDSR(DSR):
    """Goal DSR."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l2_loss = nn.MSELoss()

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info

        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            self.optimizer.zero_grad()

            goal_loss = self.goal_loss(samples_from_replay)
            goal_loss.backward()
            goal_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.goal_parameters(), self.clip_grad_norm)
            
            dsr_loss, td_abs_errors = self.dsr_loss(samples_from_replay)
            dsr_loss.backward()
            dsr_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.dsr_parameters(), self.clip_grad_norm)

            self.optimizer.step()

            # if self.prioritized_replay:
            #     self.replay_buffer.update_batch_priorities(td_abs_errors)

            # goal_param_norm = param_norm_(self.agent.goal_parameters())
            # opt_info.goalParamNorm.append(goal_grad_norm)
            # opt_info.goalParamRatio.append(goal_grad_norm / goal_param_norm)

            opt_info.goalLoss.append(goal_loss.item())
            opt_info.goalGradNorm.append(goal_grad_norm)
            opt_info.dsrLoss.append(dsr_loss.item())
            opt_info.dsrGradNorm.append(dsr_grad_norm)
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.

            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()

        self.update_itr_hyperparams(itr)
        return opt_info

    def goal_loss(self, samples):
        with torch.no_grad():
            features = self.agent.encode(samples.agent_inputs.observation)

        goal_embeddings = self.agent.embed_goal(samples.target_inputs.observation)

        loss = self.l2_loss(features, goal_embeddings)
        return loss