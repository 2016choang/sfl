
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np
import gym
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ReseedWrapper
from gym import Wrapper
from gym.spaces import Box, Discrete
from gym.wrappers.time_limit import TimeLimit
from collections import namedtuple

from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.collections import is_namedtuple_class


class GymEnvWrapper(Wrapper):

    def __init__(self, env,
            act_null_value=0, obs_null_value=0, force_float32=True, update_obs_func=None):
        super().__init__(env)
        o = self.env.reset()
        o, r, d, info = self.env.step(self.env.action_space.sample())
        env_ = self.env
        time_limit = isinstance(self.env, TimeLimit)
        while not time_limit and hasattr(env_, "env"):
            env_ = env_.env
            time_limit = isinstance(self.env, TimeLimit)
        if time_limit:
            info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
        self._time_limit = time_limit
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        build_info_tuples(info)
        self.update_obs_func = update_obs_func

    def step(self, action):
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info)
        if self.update_obs_func is not None:
            obs = self.update_obs_func(obs)
        return EnvStep(obs, r, d, info)

    def reset(self):
        obs = self.observation_space.convert(self.env.reset())
        if self.update_obs_func is not None:
            obs = self.update_obs_func(obs)
        return obs

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )





def build_info_tuples(info, name="info"):
    # Define namedtuples at module level for pickle.
    # Only place rlpyt uses pickle is in the sampler, when getting the
    # first examples, to avoid MKL threading issues...can probably turn
    # that off, (look for subprocess=True --> False), and then might
    # be able to define these directly within the class.
    ntc = globals().get(name)  # Define at module level for pickle.
    if ntc is None:
        globals()[name] = namedtuple(name, list(info.keys()))
    elif not (is_namedtuple_class(ntc) and
            sorted(ntc._fields) == sorted(list(info.keys()))):
        raise ValueError(f"Name clash in globals: {name}.")
    for k, v in info.items():
        if isinstance(v, dict):
            build_info_tuples(v, "_".join([name, k]))


def info_to_nt(value, name="info"):
    if not isinstance(value, dict):
        return value
    ntc = globals()[name]
    # Disregard unrecognized keys:
    values = {k: info_to_nt(v, "_".join([name, k]))
        for k, v in value.items() if k in ntc._fields}
    # Can catch some missing values (doesn't nest):
    values.update({k: 0 for k in ntc._fields if k not in values})
    return ntc(**values)


# To use: return a dict of keys and default values which sometimes appear in
# the wrapped env's env_info, so this env always presents those values (i.e.
# make keys and values keep the same structure and shape at all time steps.)
# Here, a dict of kwargs to be fed to `sometimes_info` should be passed as an
# env_kwarg into the `make` function, which should be used as the EnvCls.
# def sometimes_info(*args, **kwargs):
#     # e.g. Feed the env_id.
#     # Return a dictionary (possibly nested) of keys: default_values
#     # for this env.
#     return {}

class MinigridFeatureWrapper(Wrapper):
    
    def __init__(self, env, num_features=1024):
        super().__init__(env)
        self.env = env
        self.local_size = (5, 5, num_features)
        hpad, wpad = self.local_size[0] // 2 - 1, self.local_size[1] // 2 - 1
        self.feature_map = np.pad(np.random.rand(19, 19, num_features), ((hpad, hpad), (wpad, wpad), (0, 0)), mode='constant')

        self.observation_space = Box(0, 1, self.local_size)
        self.action_space = Discrete(4)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up
        self.env.unwrapped.agent_dir = action
        _, reward, done, info = self.env.step(2)
        pos = self.env.unwrapped.agent_pos
        obs = self.get_obs(pos)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()
        pos = self.env.unwrapped.agent_pos
        return self.get_obs(pos)

    def get_obs(self, pos):
        h_pos, w_pos = pos
        h_len, w_len = self.local_size[:2]
        h_len = h_len // 2
        w_len = w_len // 2
        return self.feature_map[h_pos - h_len: h_pos + h_len + 1, w_pos - w_len:w_pos + w_len + 1]


class MoveWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = Discrete(4)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up
        self.env.unwrapped.agent_dir = action
        obs, reward, done, info = self.env.step(2)
        return obs, reward, done, info


class EnvInfoWrapper(Wrapper):

    def __init__(self, env, info_example):
        super().__init__(env)
        # self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._sometimes_info = info_example

    def step(self, action):
        o, r, d, info = super().step(action)
        # Try to make info dict same key structure at every step.
        return o, r, d, infill_info(info, self._sometimes_info)


def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info


def update_obs_minigrid(obs):
    H, W = 84, 84
    resized_obs = resize(obs, (H, W), anti_aliasing=True)
    return img_as_ubyte(resized_obs)


def make(*args, info_example=None, minigrid_config=None, **kwargs):
    if minigrid_config is not None:
        mode = minigrid_config.get('mode')
        env = gym.make(*args, **kwargs)
        if minigrid_config.get('reseed', True):
            env = ReseedWrapper(env)
        if mode == 'full':
            return GymEnvWrapper(RGBImgObsWrapper(env))
        elif mode == 'small':
            env = RGBImgObsWrapper(env)
            if minigrid_config.get('move', False):
                env = MoveWrapper(env)
            return GymEnvWrapper(env, update_obs_func=update_obs_minigrid)
        elif mode == 'compact':
            return GymEnvWrapper(FullyObsWrapper(env))
        elif mode == 'random':
            return GymEnvWrapper(MinigridFeatureWrapper(RGBImgObsWrapper(env)))
    elif info_example is None:
        return GymEnvWrapper(gym.make(*args, **kwargs))
    else:
        return GymEnvWrapper(EnvInfoWrapper(
            gym.make(*args, **kwargs), info_example))
