from collections import namedtuple

import torch
import torch.nn as nn

from rlpyt.algos.dqn.dsr.dsr import DSR
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.tensor import select_at_indexes, valid_mean

OptInfo = namedtuple("OptInfo", ["dsrLoss", "dsrGradNorm", "tdAbsErr"])


class ActionDSR(DSR):
    """Action DSR."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def dsr_loss(self, samples):
        """Samples have leading batch dimension [B,..] (but not time)."""
        # 1a. encode observations in feature space
        with torch.no_grad():
            features = self.agent.encode(samples.agent_inputs.observation)
            features = select_at_indexes(samples.action[:, 0], features)

        # 1b. estimate successor features given features
        s_features = self.agent(features)

        with torch.no_grad():
            # 2a. encode target observations in feature space
            target_features = self.agent.encode(samples.target_inputs.observation)
            next_a = torch.randint(high=target_features.shape[1], size=samples.action[:, 0].shape)
            target_features = select_at_indexes(next_a, target_features)

            # 2b. estimate target successor features given features
            target_s_features = self.agent.target(target_features)

        # 3. combine current features + discounted target successor features
        disc_target_s_features = (self.discount ** self.n_step_return) * target_s_features
        y = features + (1 - samples.done_n.float()).view(-1, 1) * disc_target_s_features

        delta = y - s_features
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)

        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        # if self.prioritized_replay:
        #     losses *= samples.is_weights

        # sum losses over feature vector such that each sample has a scalar loss (result: B x 1)
        # losses = losses.sum(dim=1)

        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        if not self.mid_batch_reset:
            valid = valid_from_done(samples.done)
            loss = valid_mean(losses, valid)
            td_abs_errors *= valid
        else:
            loss = torch.mean(losses)
        return loss, td_abs_errors
