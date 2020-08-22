from collections import defaultdict, namedtuple

import torch
import torch.nn as nn

from rlpyt.algos.dqn.dsr.dsr import DSR, OptInfo
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer, LandmarkUniformReplayBuffer, VizDoomLandmarkUniformReplayBuffer,
    UniformTripletReplayBuffer)
from rlpyt.utils.buffer import torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger
from rlpyt.utils.misc import param_norm_
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done

FeatureOptInfo = namedtuple("FeatureOptInfo", ["featureLoss", "featureGradNorm",
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
    
    def append_dsr_samples(self, samples=None):
        # Append samples to replay buffer used for training successor features
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
    
    def optimize_agent(self, itr, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
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

VizDoomSamplesToBuffer = namedarraytuple("VizDoomSamplesToBuffer",
    ["observation", "action", "reward", "done", "mode", "position"])

class PositionPrediction(FeatureDSR):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l2_loss = nn.MSELoss()
        self.opt_info_class = FeatureOptInfo
        self.opt_info_fields = tuple(f for f in self.opt_info_class._fields)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        obs = examples["observation"]
        obs_pyt = torchify_buffer(obs)
        feature = self.agent.encode(obs_pyt).squeeze(0).cpu()
        example_to_buffer = VizDoomSamplesToBuffer(
            observation=feature,
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            mode=examples["agent_info"].mode,
            position=examples["env_info"].position
        )

        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        self.replay_buffer = VizDoomLandmarkUniformReplayBuffer(**replay_kwargs)
        self.feature_replay_buffer = None
        
    def samples_to_buffer(self, samples):
        feature = self.agent.encode(samples.env.observation).cpu()
        return VizDoomSamplesToBuffer(
            observation=feature,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            mode=samples.agent.agent_info.mode,
            position=samples.env.env_info.position
        )
    
    def feature_loss(self, samples):
        pred_position = self.agent.encode(samples.agent_inputs.observation, mode='output')
        loss = self.l2_loss(pred_position, samples.position.float())
        return loss, {}


TCFOptInfo = namedtuple("TCFOptInfo", ["posDistance", "negDistance"] + list(FeatureOptInfo._fields))
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "mode"])

class LandmarkTCFDSR(FeatureDSR):
    """Time Contrastive Features DSR."""

    def __init__(
        self,
        pos_threshold=3,
        neg_close_threshold=15,
        neg_far_threshold=30,
        margin=2.0,
        metric='L2',
        fixed_features=False,
        **kwargs):
        save__init__args(locals())
        super().__init__(**kwargs)
        self.opt_info_class = TCFOptInfo
        self.opt_info_fields = tuple(f for f in self.opt_info_class._fields)

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        super().initialize(agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size, rank)
        self.initialize_triplet_replay_buffer(examples, batch_spec)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        observation = examples["observation"]
        if self.fixed_features:
            obs_pyt = torchify_buffer(observation)
            observation = self.agent.encode(obs_pyt).squeeze(0).cpu()
        example_to_buffer = SamplesToBuffer(
            observation=observation,
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            mode=examples["agent_info"].mode,
        )

        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        # if self.prioritized_replay:
        #     replay_kwargs.update(dict(
        #         alpha=self.pri_alpha,
        #         beta=self.pri_beta_init,
        #         default_priority=self.default_priority,
        #     ))
        #     ReplayCls = (AsyncPrioritizedReplayFrameBuffer if async_ else
        #         PrioritizedReplayFrameBuffer)
        # else:
        # ReplayCls = (AsyncUniformReplayFrameBuffer if async_ else
        #     UniformReplayFrameBuffer)
        ReplayCls = (AsyncUniformReplayBuffer if async_ else
            LandmarkUniformReplayBuffer)
        self.replay_buffer = ReplayCls(**replay_kwargs)
        self.feature_replay_buffer = None
        self.test_replay_buffer = None

    def initialize_triplet_replay_buffer(self, examples, batch_spec, async_=False):
        observation = examples["observation"]
        if self.fixed_features:
            obs_pyt = torchify_buffer(observation)
            observation = self.agent.encode(obs_pyt).squeeze(0).cpu()
        example_to_buffer = SamplesToBuffer(
            observation=observation,
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            mode=examples["agent_info"].mode,
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
        self.test_feature_replay_buffer = UniformTripletReplayBuffer(**triplet_replay_kwargs)

    def append_test_feature_samples(self, samples=None):
        # Append samples to replay buffer used for training feature representation only
        if samples is not None and self.test_feature_replay_buffer is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.test_feature_replay_buffer.append_samples(samples_to_buffer)

    def samples_to_buffer(self, samples):
        observation = samples.env.observation
        if self.fixed_features:
            observation = self.agent.encode(observation).cpu()
        return SamplesToBuffer(
            observation=observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            mode=samples.agent.agent_info.mode
        )

    def feature_loss(self, samples):
        # Time contrastive loss
        mode = 'encode'
        if self.fixed_features:
            mode = 'output'
        anchor_embeddings = self.agent.encode(samples.anchor, mode=mode)
        pos_embeddings = self.agent.encode(samples.pos, mode=mode)
        neg_embeddings = self.agent.encode(samples.neg, mode=mode)

        if self.metric == 'L2':
            pos_dist = torch.norm(anchor_embeddings - pos_embeddings, p=2, dim=1)
            neg_dist = torch.norm(anchor_embeddings - neg_embeddings, p=2, dim=1)
            loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0).mean()
        elif self.metric == 'cos':
            pos_dist = torch.sum(anchor_embeddings * pos_embeddings, dim=1)
            neg_dist = torch.sum(anchor_embeddings * neg_embeddings, dim=1)
            loss = torch.clamp(self.margin + neg_dist - pos_dist, min=0.0).mean()
        else:
            raise NotImplementedError

        with torch.no_grad():
            feature_opt_info = {"posDistance": pos_dist.mean().item(),
                                "negDistance": neg_dist.mean().item()}

        return loss, feature_opt_info
    
    def create_test_set(self, size):
        self.test_set_size = size
        self.test_samples = self.test_feature_replay_buffer.sample_batch(size)
    
    @torch.no_grad()
    def eval_feature_loss(self):
        batches = self.test_set_size // self.batch_size
        eval_losses = []
        eval_opt_info = defaultdict(list)
        for i in range(batches):
            test_batch_samples = self.test_samples[i * self.batch_size: (i + 1) * self.batch_size]
            batch_loss, batch_opt_info = self.feature_loss(test_batch_samples)
            eval_losses.append(batch_loss)
            for key, value in batch_opt_info.items(): 
                eval_opt_info[key].append(value)
        return eval_losses, eval_opt_info

    def dsr_loss(self, samples):
        """Samples have leading batch dimension [B,..] (but not time)."""
        # 1a. encode observations in feature space
        with torch.no_grad():
            features = self.agent.encode(samples.agent_inputs.observation)

        # 1b. estimate successor features given features
        dsr = self.agent(features)
        s_features = select_at_indexes(samples.action, dsr)

        with torch.no_grad():
            # 2a. encode target observations in feature space
            target_features = self.agent.encode(samples.target_inputs.observation)

            # 2b. estimate target successor features given features
            target_dsr = self.agent.target(target_features)

            # next_qs = self.agent.q_estimate(target_dsr)
            # next_a = torch.argmax(next_qs, dim=-1)
            # random actions
            next_a = torch.randint(high=target_dsr.shape[1], size=samples.action.shape)

            target_s_features = select_at_indexes(next_a, target_dsr)

        # 3. combine current features + discounted target successor features
        done_n = samples.done_n.float().view(-1, 1)
        disc_target_s_features = (self.discount ** self.n_step_return) * target_s_features
        s_y = target_features + (1 - samples.target_done.float()).view(-1, 1) * disc_target_s_features
        y = features * done_n + (1 - done_n) * s_y

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

        td_abs_errors = abs_delta.mean(axis=1).detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        if not self.mid_batch_reset:
            losses = torch.mean(losses, axis=1)
            valid = valid_from_done(samples.done)
            loss = valid_mean(losses, valid)
            td_abs_errors *= valid
        else:
            loss = torch.mean(losses)
        return loss, td_abs_errors

SFeatureOptInfo = namedtuple("SFeatureOptInfo", ["dsrLoss", "dsrGradNorm", "percentDSRLoss", "tdAbsErr", "featureNorm",
                                                 "targetFeatureNorm", "dsrNorm", "targetDSRNorm"])

class FixedFeatureDSR(DSR):
    """Fixed feature DSR."""

    def __init__(
            self,
            min_steps_dsr_learn=int(5e4),
            scale_target=True,
            **kwargs):
        super().__init__(**kwargs)
        save__init__args(locals())
        self.scale_target_factor = 1 * (1 - (self.discount ** self.n_step_return)) / (1 - self.discount)
        self.initial_dsr_loss = None
    
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
        if self.initial_optim_state_dict is not None:
            self.dsr_optimizer.load_state_dict(self.initial_optim_state_dict['dsr'])
        # if self.prioritized_replay:
        #     self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

    def samples_to_buffer(self, samples):
        feature = self.agent.encode(samples.env.observation).cpu()
        return SamplesToBuffer(
            observation=feature,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            mode=samples.agent.agent_info.mode
        )
    
    def example_to_buffer(self, examples):
        obs = examples["observation"]
        obs_pyt = torchify_buffer(obs)
        feature = self.agent.encode(obs_pyt).squeeze(0).cpu()
        return SamplesToBuffer(
            observation=feature,
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            mode=examples["agent_info"].mode,
        )

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = self.example_to_buffer(examples)

        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        # if self.prioritized_replay:
        #     replay_kwargs.update(dict(
        #         alpha=self.pri_alpha,
        #         beta=self.pri_beta_init,
        #         default_priority=self.default_priority,
        #     ))
        #     ReplayCls = (AsyncPrioritizedReplayFrameBuffer if async_ else
        #         PrioritizedReplayFrameBuffer)
        # else:
        # ReplayCls = (AsyncUniformReplayFrameBuffer if async_ else
        #     UniformReplayFrameBuffer)
        ReplayCls = (AsyncUniformReplayBuffer if async_ else
            LandmarkUniformReplayBuffer)
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optim_state_dict(self):
        """If carrying multiple optimizers, overwrite to return dict state_dicts."""
        return {'dsr': self.dsr_optimizer.state_dict()}
    
    def append_feature_samples(self, samples=None):
        pass

    def append_dsr_samples(self, samples=None):
        # Append samples to replay buffer used for training successor features
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
    
    def optimize_agent(self, itr, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        opt_info = SFeatureOptInfo(*([] for _ in range(len(SFeatureOptInfo._fields))))
        feature_info = {}
        if itr < self.min_itr_learn:
            # Not enough samples have been collected
            return opt_info, feature_info

        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)

            if itr >= self.min_itr_dsr_learn:
                # Train successor feature representation
                self.dsr_optimizer.zero_grad()

                dsr_loss, td_abs_errors, feature_info, norm_info = self.dsr_loss(samples_from_replay)
                if self.initial_dsr_loss is None:
                    self.initial_dsr_loss = dsr_loss.item()
                dsr_loss.backward()
                dsr_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.dsr_parameters(), self.clip_grad_norm)

                self.dsr_optimizer.step()

                opt_info.dsrLoss.append(dsr_loss.item())
                opt_info.dsrGradNorm.append(dsr_grad_norm)
                opt_info.percentDSRLoss.append(dsr_loss.item() / self.initial_dsr_loss)
                for key, value in norm_info.items():
                    getattr(opt_info, key).append(value.item())
                opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.

                self.update_counter += 1
                if self.update_counter % self.target_update_interval == 0:
                    self.agent.update_target()
        
        self.update_itr_hyperparams(itr)
        return opt_info, feature_info

    def dsr_loss(self, samples):
        """Samples have leading batch dimension [B,..] (but not time)."""

        feature_info = {}
        norm_info = {}

        # 1a. encode observations in feature space
        with torch.no_grad():
            features = samples.agent_inputs.observation
            features_n = samples.observation_n
            if self.scale_target:
                features_n = features_n / self.scale_target_factor
            feature_info['singleFeature'] = features[:, 0]
            feature_info['singleFeatureN'] = features_n[:, 0]
            feature_info['feature'] = features
            feature_info['featureN'] = features_n
            norm_info['featureNorm'] = torch.norm(features, p=2, dim=-1).mean()
            norm_info['targetFeatureNorm'] = torch.norm(features_n, p=2, dim=-1).mean()

        # 1b. estimate successor features given features
        dsr = self.agent(features)
        s_features = select_at_indexes(samples.action, dsr)
        feature_info['SFeature'] = s_features
        feature_info['singleSFeature'] = s_features[:, 0]
        norm_info['dsrNorm'] = torch.norm(s_features, p=2, dim=-1).mean()

        with torch.no_grad():
            # 2a. encode target observations in feature space
            target_features = samples.target_inputs.observation

            # 2b. estimate target successor features given features
            target_dsr = self.agent.target(target_features)

            # next_qs = self.agent.q_estimate(target_dsr)
            # next_a = torch.argmax(next_qs, dim=-1)
            # random actions
            # next_a = torch.randint(high=target_dsr.shape[1], size=samples.action.shape)
            # next_a = torch.zeros_like(samples.action)

            # target_s_features = select_at_indexes(next_a, target_dsr)
            target_s_features = torch.mean(target_dsr, dim=1)
            feature_info['targetSFeature'] = target_s_features
            feature_info['singleTargetSFeature'] = target_s_features[:, 0]
            norm_info['targetDSRNorm'] = torch.norm(target_s_features, p=2, dim=-1).mean()

        # 3. combine current features + discounted target successor features
        disc_target_s_features = (self.discount ** self.n_step_return) * target_s_features
        y = features_n + (1 - samples.done_n.float().view(-1, 1)) * disc_target_s_features 

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

        td_abs_errors = abs_delta.mean(axis=1).detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        if not self.mid_batch_reset:
            losses = torch.mean(losses, axis=1)
            valid = valid_from_done(samples.done)
            loss = valid_mean(losses, valid)
            td_abs_errors *= valid
        else:
            loss = torch.mean(losses)
        return loss, td_abs_errors, feature_info, norm_info
