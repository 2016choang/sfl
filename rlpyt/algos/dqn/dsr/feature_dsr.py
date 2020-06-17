from collections import namedtuple

import torch
import torch.nn as nn

from rlpyt.algos.dqn.dsr.dsr import DSR, SamplesToBuffer
from rlpyt.replays.non_sequence.uniform import UniformTripletReplayBuffer
from rlpyt.utils.logging import logger
from rlpyt.utils.misc import param_norm_
from rlpyt.utils.quick_args import save__init__args

FeatureOptInfo = namedtuple("FeateureOptInfo", ["featureLoss", "featureGradNorm",
                                                "dsrLoss", "dsrGradNorm", "tdAbsErr"])

class FeatureDSR(DSR):
    """Feature-based DSR."""

    def __init__(
            self,
            feature_learning_rate=2.5e4,
            max_steps_feature_learn=None,
            min_steps_dsr_learn=int(5e4),
            **kwargs):
        super().__init__(**kwargs)
        save__init__args(locals())
    
    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        super().initialize(agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size, rank)
        if self.max_steps_feature_learn is not None:
            self.max_itr_feature_learn = int(self.max_steps_feature_learn // self.sampler_bs)
        else:
            self.max_itr_feature_learn = None
        self.min_itr_dsr_learn = int(self.min_steps_dsr_learn // self.sampler_bs)

    def optim_initialize(self, rank=0):
        """Called by async runner."""
        self.rank = rank
        self.dsr_optimizer = self.OptimCls(self.agent.dsr_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.feature_optimizer = self.OptimCls(self.agent.feature_parameters(),
            lr=self.feature_learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.dsr_optimizer.load_state_dict(self.initial_optim_state_dict['dsr'])
            self.feature_optimizer.load_state_dict(self.initial_optim_state_dict['feature'])
        # if self.prioritized_replay:
        #     self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)
    
    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        super().initialize_replay_buffer(examples, batch_spec, async_)
        self.feature_replay_buffer = None

    def optim_state_dict(self):
        """If carrying multiple optimizers, overwrite to return dict state_dicts."""
        return {'dsr': self.dsr_optimizer.state_dict(),
                'feature': self.feature_optimizer.state_dict()}
    
    def append_feature_samples(self, samples=None):
        # Append samples to replay buffer used for training feature representation only
        if samples is not None and self.feature_replay_buffer is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.feature_replay_buffer.append_samples(samples_to_buffer)
        
    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            # Append samples to replay buffer used for training successor features
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = self.opt_info_class(*([] for _ in range(len(self.opt_info_class._fields))))
        if itr < self.min_itr_learn:
            # Not enough samples have been collected
            return opt_info

        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            if self.max_itr_feature_learn is None or itr < self.max_itr_feature_learn: 
                # Train feature representation
                if self.feature_replay_buffer:
                    feature_samples_from_replay = self.feature_replay_buffer.sample_batch(self.batch_size)
                else:
                    feature_samples_from_replay = samples_from_replay

                self.feature_optimizer.zero_grad()

                feature_loss, feature_opt_info = self.feature_loss(feature_samples_from_replay)
                feature_loss.backward()
                feature_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.feature_parameters(), self.clip_grad_norm)

                self.feature_optimizer.step()

                opt_info.featureLoss.append(feature_loss.item())
                opt_info.featureGradNorm.append(feature_grad_norm)
                for key, value in feature_opt_info.items():
                    getattr(opt_info, key).append(value)

            if itr >= self.min_itr_dsr_learn:
                # Train successor feature representation
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

    def feature_loss(self, samples_from_replay):
        raise NotImplementedError


IDFOptInfo = namedtuple("IDFOptInfo", ["idfAccuracy"] + list(FeatureOptInfo._fields))

class IDFDSR(FeatureDSR):
    """Inverse Dynamics Features DSR."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opt_info_class = IDFOptInfo
        self.opt_info_fields = tuple(f for f in self.opt_info_class._fields)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def feature_loss(self, samples):
        # Inverse dynamics prediction loss
        pred_actions = self.agent.inverse_dynamics(samples.agent_inputs.observation,
                                            samples.target_inputs.observation)

        loss = self.cross_entropy_loss(pred_actions, samples.action)
        with torch.no_grad():
            accuracy = ((pred_actions.argmax(dim=1) == samples.action).sum().float() / samples.action.shape[0]) * 100
            feature_opt_info = {'idfAccuracy': accuracy.item()}
        return loss, feature_opt_info


TCFOptInfo = namedtuple("TCFOptInfo", ["posDistance", "negDistance"] + list(FeatureOptInfo._fields))

class TCFDSR(FeatureDSR):
    """Time Contrastive Features DSR."""

    def __init__(
        self,
        pos_threshold=3,
        neg_close_threshold=15,
        neg_far_threshold=30,
        margin=2.0,
        **kwargs):
        save__init__args(locals())
        super().__init__(**kwargs)
        self.opt_info_class = TCFOptInfo
        self.opt_info_fields = tuple(f for f in self.opt_info_class._fields)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        super().initialize_replay_buffer(examples, batch_spec, async_)
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )

        triplet_replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            pos_threshold=self.pos_threshold,
            neg_close_threshold=self.neg_close_threshold,
            neg_far_threshold=self.neg_far_threshold
        )

        self.feature_replay_buffer = UniformTripletReplayBuffer(**triplet_replay_kwargs)

    def feature_loss(self, samples):
        # Time contrastive loss
        anchor_embeddings = self.agent.encode(samples.anchor)
        pos_embeddings = self.agent.encode(samples.pos)
        neg_embeddings = self.agent.encode(samples.neg)

        pos_dist = torch.norm(anchor_embeddings - pos_embeddings, p=2, dim=1)
        neg_dist = torch.norm(anchor_embeddings - neg_embeddings, p=2, dim=1)

        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0).mean()
        with torch.no_grad():
            feature_opt_info = {"posDistance": pos_dist.mean().item(),
                                "negDistance": neg_dist.mean().item()}
        return loss, feature_opt_info
