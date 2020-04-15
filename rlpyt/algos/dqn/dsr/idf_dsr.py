from collections import namedtuple

import torch
import torch.nn as nn

from rlpyt.algos.dqn.dsr.dsr import DSR
from rlpyt.utils.logging import logger
from rlpyt.utils.misc import param_norm_
from rlpyt.utils.quick_args import save__init__args

OptInfo = namedtuple("OptInfo", ["idfLoss", "idfAccuracy", "idfGradNorm",
                                 "dsrLoss", "dsrGradNorm", "tdAbsErr"])


class IDFDSR(DSR):
    """Inverse Dynamics Features DSR."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            idf_learning_rate=2.5e-4,
            idf_update_interval=312,  # 312 * 32 = 1e4 env steps.
            min_steps_dsr_learn=int(5e4),
            **kwargs):
        super().__init__(**kwargs)
        save__init__args(locals())
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.idf_update_counter = 0
        self.idf_update = True
    
    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        super().initialize(agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size, rank)
        self.min_itr_dsr_learn = int(self.min_steps_dsr_learn // self.sampler_bs)

    def optim_initialize(self, rank=0):
        """Called by async runner."""
        self.rank = rank
        self.dsr_optimizer = self.OptimCls(self.agent.dsr_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.idf_optimizer = self.OptimCls(self.agent.idf_parameters(),
            lr=self.idf_learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.dsr_optimizer.load_state_dict(self.initial_optim_state_dict['dsr'])
            self.idf_optimizer.load_state_dict(self.initial_optim_state_dict['idf'])
        # if self.prioritized_replay:
        #     self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

    def optim_state_dict(self):
        """If carrying multiple optimizers, overwrite to return dict state_dicts."""
        return {'dsr': self.dsr_optimizer.state_dict(),
                'idf': self.idf_optimizer.state_dict()}

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
            self.idf_optimizer.zero_grad()

            idf_loss, accuracy = self.idf_loss(samples_from_replay)
            idf_loss.backward()
            idf_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.idf_parameters(), self.clip_grad_norm)

            self.idf_optimizer.step()

            opt_info.idfLoss.append(idf_loss.item())
            opt_info.idfAccuracy.append(accuracy.item())
            opt_info.idfGradNorm.append(idf_grad_norm)

            if itr >= self.min_itr_dsr_learn:
                self.dsr_optimizer.zero_grad()

                dsr_loss, td_abs_errors = self.dsr_loss(samples_from_replay)
                dsr_loss.backward()
                dsr_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.dsr_parameters(), self.clip_grad_norm)

                self.dsr_optimizer.step()

                opt_info.dsrLoss.append(dsr_loss.item())
                opt_info.dsrGradNorm.append(dsr_grad_norm)
                opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.

                self.update_counter += 1
                if self.update_counter % self.target_update_interval == 0:
                    self.agent.update_target()
            
        self.update_itr_hyperparams(itr)
        return opt_info

    def idf_loss(self, samples):
        pred_actions = self.agent.inverse_dynamics(samples.agent_inputs.observation,
                                                   samples.target_inputs.observation)

        loss = self.cross_entropy_loss(pred_actions, samples.action)
        with torch.no_grad():
            accuracy = ((pred_actions.argmax(dim=1) == samples.action).sum().float() / samples.action.shape[0]) * 100
        return loss, accuracy