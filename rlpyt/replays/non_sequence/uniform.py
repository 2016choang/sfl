
import math
import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.algos.utils import discount_return_n_step
from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims, torchify_buffer
from rlpyt.utils.collections import namedarraytuple


TripletsFromReplay = namedarraytuple("TripletsFromReplay",
    ["anchor", "pos", "neg"])

class UniformReplay:

    def sample_batch(self, batch_B):
        T_idxs, B_idxs = self.sample_idxs(batch_B)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_B):
        t, b, f = self.t, self.off_backward, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b
        low = 0 if self._buffer_full else f
        T_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f  # min for invalid high t.
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs


class UniformReplayBuffer(UniformReplay, NStepReturnBuffer):
    pass


class AsyncUniformReplayBuffer(AsyncReplayBufferMixin, UniformReplayBuffer):
    pass

LandmarkSamplesFromReplay = namedarraytuple("LandmarkSamplesFromReplay",
    ["agent_inputs", "action", "return_", "done", "done_n", "target_inputs", "target_done"])

class LandmarkUniformReplayBuffer(UniformReplayBuffer):
    
    def __init__(self, example, size, B, discount=False,**kwargs):
        super().__init__(example, size, B, **kwargs)
        self.discount = discount
        if discount:
            self.samples_observation_n = buffer_from_example(example.observation, (self.T, self.B),
                share_memory=self.async_)
        else:
            self.samples_observation_n = self.samples.observation
        self.valid_idxs = np.arange(self.T, dtype=int)
    
    def compute_returns(self, T):
        """e.g. if 2-step return, t-1 is first return written here, using reward
        at t-1 and new reward at t (up through t-1+T from t+T)."""
        if self.n_step_return == 1:
            return  # return = reward, done_n = done
        t, s = self.t, self.samples
        nm1 = self.n_step_return - 1
        if t - nm1 >= 0 and t + T <= self.T:  # No wrap (operate in-place).
            reward = s.reward[t - nm1:t + T]
            done = s.done[t - nm1:t + T]
            return_dest = self.samples_return_[t - nm1: t - nm1 + T]
            done_n_dest = self.samples_done_n[t - nm1: t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)

            observation = s.observation[t - nm1: t + T]
            observation_dest = self.samples_observation_n[t - nm1: t - nm1 + T]
            discount_return_n_step(observation, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=observation_dest)

        else:  # Wrap (copies); Let it (wrongly) wrap at first call.
            idxs = np.arange(t - nm1, t + T) % self.T
            reward = s.reward[idxs]
            done = s.done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            self.samples_return_[dest_idxs] = return_
            self.samples_done_n[dest_idxs] = done_n

            observation = s.observation[idxs]
            observation_, done_n = discount_return_n_step(observation, done,
                n_step=self.n_step_return, discount=self.discount)
            self.samples_observation_n[dest_idxs] = observation_

    def sample_idxs(self, batch_B):
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        t, b, f = self.t, self.off_backward, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b
        low = 0 if self._buffer_full else f

        T_idxs = np.zeros((batch_B, ), dtype=int)

        if self._buffer_full:
            saved = self.samples.mode[t - b:t - b + min(t, b) + f].copy()
            self.samples.mode[t - b:t - b + min(t, b) + f] = True

        for i, B_idx in enumerate(B_idxs):
            mask = self.samples.mode[low:high, B_idx]
            T_idxs[i] = np.random.choice(self.valid_idxs[low:high][~mask])

        if self._buffer_full and 0 < min(t, b) + f:
            self.samples.mode[t - b:t - b + min(t, b) + f] = saved

        return T_idxs, B_idxs

    def extract_batch(self, T_idxs, B_idxs):
        s = self.samples
        target_T_idxs = (T_idxs + self.n_step_return) % self.T
        batch = LandmarkSamplesFromReplay(
            agent_inputs=AgentInputs(
                observation=self.samples_observation_n[T_idxs, B_idxs],
                prev_action=s.action[T_idxs - 1, B_idxs],
                prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            return_=self.samples_return_[T_idxs, B_idxs],
            done=self.samples.done[T_idxs, B_idxs],
            done_n=self.samples_done_n[T_idxs, B_idxs],
            target_inputs=AgentInputs(
                observation=self.extract_observation(target_T_idxs, B_idxs),
                prev_action=s.action[target_T_idxs - 1, B_idxs],
                prev_reward=s.reward[target_T_idxs - 1, B_idxs],
            ),
            target_done=self.samples.done[target_T_idxs, B_idxs],
        )
        t_news = np.where(s.done[T_idxs - 1, B_idxs])[0]
        batch.agent_inputs.prev_action[t_news] = 0
        batch.agent_inputs.prev_reward[t_news] = 0
        return torchify_buffer(batch)

VizDoomLandmarkSamplesFromReplay = namedarraytuple("VizDoomLandmarkSamplesFromReplay",
    ["agent_inputs", "action", "return_", "done", "done_n", "position", "target_inputs", "target_done"])

class VizDoomLandmarkUniformReplayBuffer(LandmarkUniformReplayBuffer):

    def extract_batch(self, T_idxs, B_idxs):
        s = self.samples
        target_T_idxs = (T_idxs + self.n_step_return) % self.T
        batch = VizDoomLandmarkSamplesFromReplay(
            agent_inputs=AgentInputs(
                observation=self.samples_observation_n[T_idxs, B_idxs],
                prev_action=s.action[T_idxs - 1, B_idxs],
                prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            return_=self.samples_return_[T_idxs, B_idxs],
            done=self.samples.done[T_idxs, B_idxs],
            done_n=self.samples_done_n[T_idxs, B_idxs],
            position=self.samples.position[T_idxs, B_idxs],
            target_inputs=AgentInputs(
                observation=self.extract_observation(target_T_idxs, B_idxs),
                prev_action=s.action[target_T_idxs - 1, B_idxs],
                prev_reward=s.reward[target_T_idxs - 1, B_idxs],
            ),
            target_done=self.samples.done[target_T_idxs, B_idxs],
        )
        t_news = np.where(s.done[T_idxs - 1, B_idxs])[0]
        batch.agent_inputs.prev_action[t_news] = 0
        batch.agent_inputs.prev_reward[t_news] = 0
        return torchify_buffer(batch)
    

class UniformTripletReplayBuffer(BaseReplayBuffer):
    
    def __init__(self, example, size, B, pos_threshold, neg_close_threshold, neg_far_threshold):
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.pos_threshold = pos_threshold
        self.neg_close_threshold = neg_close_threshold
        self.neg_far_threshold = neg_far_threshold
        self.t = 0  # Cursor (in T dimension).
        self.samples = buffer_from_example(example, (T, B),
            share_memory=self.async_)
        self.episode_bounds = np.zeros((T, B, 2), dtype=int)
        self.episode_bounds[:, :, 0] = -self.T
        self.episode_bounds[:, :, 1] = self.T
        self.episode_start = 0
        self._buffer_full = False

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        t = self.t
        if t + T > self.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = np.arange(t, t + T)
        self.samples[idxs] = samples
        if self.episode_start >= t:
            bounds = [self.episode_start - self.T, t]
        else:
            bounds = [self.episode_start, t]
        self.episode_bounds[idxs] = bounds 

        done = samples.done.detach().numpy()
        any_done = np.any(done, axis=1)

        for done_idx, done_markers in zip(idxs[any_done], done[any_done]):
            if self.episode_start >= done_idx + 1:
                bounds = [self.episode_start - self.T, done_idx + 1]
            else:
                bounds = [self.episode_start, done_idx + 1]

            self.episode_bounds[self.episode_start: done_idx + 1, done_markers] = bounds
            self.episode_start = done_idx + 1

        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.T
        return T, idxs  # Pass these on to subclass.

    def sample_batch(self, batch_B):
        t = self.t
        high = self.T if self._buffer_full else t - self.neg_close_threshold
        low = 0 if self._buffer_full else self.neg_close_threshold
        anchor_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        # anchor_idxs[anchor_idxs >= t] += t # min for invalid high t.
        anchor_idxs = anchor_idxs % self.T
        
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))

        pos_low = np.maximum(anchor_idxs - self.pos_threshold, self.episode_bounds[anchor_idxs, B_idxs, 0])
        upper_bounds = self.episode_bounds[anchor_idxs, B_idxs, 1]
        invalid_bounds = anchor_idxs >= upper_bounds
        upper_bounds[invalid_bounds] += self.T
        pos_high = np.minimum(anchor_idxs + self.pos_threshold + 1, upper_bounds)
        upper_bounds[invalid_bounds] -= self.T

        pos_idxs = np.random.randint(low=pos_low, high=pos_high, size=(batch_B,))
        # pos_idxs = pos_idxs % self.T
        # pos_idxs[pos_idxs >= t] += t
        pos_idxs = pos_idxs % self.T

        left_neg_low = np.maximum(anchor_idxs - self.neg_far_threshold, self.episode_bounds[anchor_idxs, B_idxs, 0])
        left_neg_high = anchor_idxs - self.neg_close_threshold + 1
        invalid = left_neg_low >= left_neg_high
        left_neg_low[invalid] = left_neg_high[invalid] - 1
        left_neg_idxs = np.random.randint(low=left_neg_low, high=left_neg_high, size=(batch_B,))
        left_range = left_neg_high - left_neg_low
        left_range[invalid] = 0

        right_neg_low = anchor_idxs + self.neg_close_threshold
        right_neg_high = np.minimum(anchor_idxs + self.neg_far_threshold + 1, self.episode_bounds[anchor_idxs, B_idxs, 1])
        invalid = right_neg_low >= right_neg_high
        right_neg_low[invalid] = right_neg_high[invalid] - 1
        right_neg_idxs = np.random.randint(low=right_neg_low, high=right_neg_high, size=(batch_B,))
        right_range = right_neg_high - right_neg_low
        right_range[invalid] = 0

        prob = left_range / np.clip(left_range + right_range, a_min=1, a_max=None)
        uniform = np.random.rand(*prob.shape)
        neg_idxs = np.where(uniform < prob, left_neg_idxs, right_neg_idxs)
        # neg_idxs = neg_idxs % self.T
        # neg_idxs[neg_idxs >= t] += t
        neg_idxs = neg_idxs % self.T

        batch = TripletsFromReplay(
            anchor=self.extract_observation(anchor_idxs, B_idxs),
            pos=self.extract_observation(pos_idxs, B_idxs),
            neg=self.extract_observation(neg_idxs, B_idxs)
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs):
        return self.samples.observation[T_idxs, B_idxs]
