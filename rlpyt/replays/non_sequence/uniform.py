
import math
import numpy as np

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
        self.episode_bounds = np.zeros((T, 2), dtype=int)
        self.episode_bounds[:, 0] = -self.T
        self.episode_bounds[:, 1] = self.T
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

        for done_idx in idxs[samples.done.flatten().detach().numpy()]:
            if self.episode_start >= done_idx + 1:
                bounds = [self.episode_start - self.T, done_idx + 1]
            else:
                bounds = [self.episode_start, done_idx + 1]

            self.episode_bounds[self.episode_start: done_idx + 1] = bounds
            self.episode_start = done_idx + 1

        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.T
        return T, idxs  # Pass these on to subclass.

    def sample_batch(self, batch_B):
        t = self.t
        high = self.T if self._buffer_full else t - self.neg_far_threshold
        low = 0 if self._buffer_full else self.neg_far_threshold
        anchor_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        # anchor_idxs[anchor_idxs >= t] += t # min for invalid high t.
        anchor_idxs = anchor_idxs % self.T

        pos_low = np.maximum(anchor_idxs - self.pos_threshold, self.episode_bounds[anchor_idxs][:, 0])
        upper_bounds = self.episode_bounds[anchor_idxs][:, 1]
        invalid_bounds = anchor_idxs >= upper_bounds
        upper_bounds[invalid_bounds] += self.T
        pos_high = np.minimum(anchor_idxs + self.pos_threshold + 1, upper_bounds)
        upper_bounds[invalid_bounds] -= self.T

        pos_idxs = np.random.randint(low=pos_low, high=pos_high, size=(batch_B,))
        # pos_idxs = pos_idxs % self.T
        # pos_idxs[pos_idxs >= t] += t
        pos_idxs = pos_idxs % self.T

        left_neg_low = np.maximum(anchor_idxs - self.neg_far_threshold, self.episode_bounds[anchor_idxs][:, 0])
        left_neg_high = anchor_idxs - self.neg_close_threshold + 1
        invalid = left_neg_low >= left_neg_high
        left_neg_low[invalid] = left_neg_high[invalid] - 1
        left_neg_idxs = np.random.randint(low=left_neg_low, high=left_neg_high, size=(batch_B,))
        left_range = left_neg_high - left_neg_low
        left_range[invalid] = 0

        right_neg_low = anchor_idxs + self.neg_close_threshold
        right_neg_high = np.minimum(anchor_idxs + self.neg_far_threshold + 1, self.episode_bounds[anchor_idxs][:, 1])
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

        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))

        batch = TripletsFromReplay(
            anchor=self.extract_observation(anchor_idxs, B_idxs),
            pos=self.extract_observation(pos_idxs, B_idxs),
            neg=self.extract_observation(neg_idxs, B_idxs)
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs):
        return self.samples.observation[T_idxs, B_idxs]
