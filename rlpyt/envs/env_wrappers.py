from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import multiprocessing
import sys
import traceback

import gym
import gym.spaces
import numpy as np

class LimitDuration(object):
  """End episodes after specified number of steps."""

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._step is None:
      self._step = 0
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()