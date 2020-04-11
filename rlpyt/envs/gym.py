
from collections import namedtuple
import random

from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np
import gym
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper, ReseedWrapper
from gym_minigrid.envs.multiroom import MultiRoomEnv
from gym import Wrapper
from gym.spaces import Box, Discrete, Dict
from gym.wrappers.time_limit import TimeLimit

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

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_random_start():
    if random.random() < 0.5:
        x = random.randint(1, 8)
    else:
        x = random.randint(10, 17)
    if random.random() < 0.5:
        y = random.randint(1, 8) 
    else:
        y = random.randint(10, 17)
    return np.array([x, y])


class MinigridImageWrapper(Wrapper):
    
    def __init__(self, env, size=(19, 19), grayscale=True, collection=False, terminate=False, reset_same=False, reset_episodes=1):
        super().__init__(env)
        self.env = env
        
        self.size = size
        self.grayscale = grayscale

        self.terminate = terminate

        self.reset_same = reset_same
        self.start_pos = get_random_start()
        self.reset_episodes = reset_episodes
        self.episodes = 0

        if grayscale:
            self.observation_space = Box(0, 1, size)
        else:
            self.observation_space = Box(0, 1, (*size, 3))

        self.collection = collection
        if self.collection:
            self.action_space = Discrete(5)
        else:
            self.action_space = Discrete(4)

    def step(self, action):
        # # 0 -- turn left  1 -- turn right  2 -- move forward
        # obs, reward, done, info = self.env.step(action)

        # 0 -- right, 1 -- down, 2 -- left, 3 -- up, 4 -- do nothing
        if action == 4:
            self.env.step(0)
            obs, reward, done, info = self.env.step(1)
        else:
            self.env.unwrapped.agent_dir = action
            obs, reward, done, info = self.env.step(2)

        obs = self.get_obs(obs)
        if not self.terminate:
            done = self.env.steps_remaining == 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()

        if self.reset_same or self.episodes != self.reset_episodes:
            self.env.unwrapped.agent_pos = self.start_pos
        else:
            self.start_pos = get_random_start()
            self.env.unwrapped.agent_pos = self.start_pos
            self.episodes = 0
        
        self.env.step(0)
        obs, _, _, _ = self.env.step(1)
        self.env.steps_remaining = self.env.max_steps
        
        self.episodes += 1
        return self.get_obs(obs) 

    def get_obs(self, obs):
        resized = resize(obs['image'], self.size, anti_aliasing=True)
        if self.grayscale:
            return np.dot(resized, [0.2989, 0.5870, 0.1140])
        else:
            return resized


class MinigridMultiRoomOracleWrapper(Wrapper):
    # 0 -- right, 1 -- down, 2 -- left, 3 -- up

    def __init__(self, env, epsilon=0.05):
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = Discrete(4)

        self.epsilon = epsilon
        self.option_index = 0
        self.option_path = []

        self.visited = np.zeros((self.env.grid.height, self.env.grid.width)) - 1

    def generate_path(self, start, goal):
        delta = start - goal

        path = np.zeros(sum(abs(delta)), dtype=int)
        path[:abs(delta[1])] = -1

        if delta[0] > 0:
            horizontal_act = 2
        else:
            horizontal_act = 0
        if delta[1] > 0:
            vertical_act = 3
        else:
            vertical_act = 1

        path[path == 0] = horizontal_act
        path[path == -1] = vertical_act

        return path
    
    def get_room(self, pos):
        for i, room in enumerate(self.env.rooms):
            if all(pos == room.exitDoorPos):
                return i

            top_left = np.array(room.top)
            bot_right = top_left + room.size - 1
            if all(top_left < pos) and all(pos < bot_right):
                return i
        raise RuntimeError

    def reset_room_option(self):
        self.option_index = 0
        if np.random.random() < self.epsilon:
            cur_pos = self.env.unwrapped.agent_pos
            goal_room = np.random.randint(self.env.num_rooms)
            cur_room = self.get_room(cur_pos)

            opposite = False
            if goal_room < cur_room:
                rooms = range(cur_room - 1, goal_room - 1, -1)
                opposite = True
            else:
                rooms = range(cur_room, goal_room, 1)           

            path = []
            for i in rooms:
                door = np.array(self.env.rooms[i].exitDoorPos)
                next_to_door = door.copy()
                next_pos = door.copy()
                exit_door_direction = self.exit_door_directions[i]

                if opposite:
                    exit_door_direction = (exit_door_direction + 2) % 4

                if exit_door_direction == 0:
                    next_to_door[0] -= 1
                    next_pos[0] +=1
                elif exit_door_direction == 1:
                    next_to_door[1] -= 1
                    next_pos[1] += 1
                elif exit_door_direction == 2:
                    next_to_door[0] += 1
                    next_pos[0] -= 1
                else: 
                    next_to_door[1] += 1
                    next_pos[1] -= 1

                path.extend(self.generate_path(cur_pos, next_to_door))

                path.append(exit_door_direction)
                path.append(exit_door_direction)
                cur_pos = next_pos

            self.option_path = path
        else:
            self.option_path = []

    def get_option_action(self):
        if self.option_index < len(self.option_path):
            return self.option_path[self.option_index]
        else:
            return None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.visited[tuple(self.env.unwrapped.agent_pos)] += 1

        self.option_index += 1
        if self.option_index >= len(self.option_path):
            self.reset_room_option()

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()
        org_dir = self.env.unwrapped.agent_dir
        org_pos = self.env.unwrapped.agent_pos.copy()

        self.exit_door_directions = []
        for room in self.env.rooms[:-1]:
            top_left = np.array(room.top)
            bot_right = top_left + room.size - 1

            exit_door = np.array(room.exitDoorPos)
            open_door_pos = exit_door.copy()

            if exit_door[0] == top_left[0]:
                exit_door_direction = 2
                open_door_pos[0] += 1
            elif exit_door[0] == bot_right[0]:
                exit_door_direction = 0
                open_door_pos[0] -= 1
            elif exit_door[1] == top_left[1]:
                exit_door_direction = 3
                open_door_pos[1] += 1
            elif exit_door[1] == bot_right[1]:
                exit_door_direction = 1
                open_door_pos[1] -= 1
            else:
                raise RuntimeError

            self.env.env.unwrapped.agent_pos = open_door_pos
            self.env.env.unwrapped.agent_dir = exit_door_direction
            self.env.step(4)

            self.exit_door_directions.append(exit_door_direction)

        self.reset_room_option()
        self.env.unwrapped.agent_dir = org_dir
        self.env.unwrapped.agent_pos = org_pos
        
        self.visited[tuple(self.env.unwrapped.agent_pos)] += 1

        obs = self.env.step(5)[0]
        self.env.steps_remaining = self.env.max_steps
        return obs


class MinigridMultiRoomWrapper(Wrapper):
    
    def __init__(self, env, num_rooms=10, size=(25, 25), grayscale=True, terminate=False, reset_same=False, reset_episodes=1):
        super().__init__(env)
        self.num_rooms = num_rooms
        self.env = env
        
        self.size = size
        self.grayscale = grayscale

        self.terminate = terminate

        self.reset_same = reset_same
        self.start_pos = None
        self.reset_episodes = reset_episodes
        self.episodes = 0

        if grayscale:
            self.observation_space = Box(0, 1, size)
        else:
            self.observation_space = Box(0, 1, (*size, 3))

        self.action_space = Discrete(6)

    def get_random_room_start(self):
        room = self.env.rooms[np.random.randint(self.num_rooms)]
        return self.env.place_agent(room.top, room.size)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up, 4 - toggle doors, 5 - do nothing
        if action == 4:
            obs, reward, done, info = self.env.step(5)
        elif action == 5:
            obs, reward, done, info = self.env.step(6)
        else:
            self.env.unwrapped.agent_dir = action
            obs, reward, done, info = self.env.step(2)

        obs = self.get_obs(obs)
        if not self.terminate:
            done = self.env.steps_remaining == 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()

        if self.reset_same or self.episodes != self.reset_episodes:
            if self.start_pos is None:
                self.start_pos = self.get_random_room_start()
            self.env.unwrapped.agent_pos = self.start_pos
        else:
            self.start_pos = self.get_random_room_start()
            self.env.unwrapped.agent_pos = self.start_pos
            self.episodes = 0
        
        self.env.step(0)
        obs, _, _, _ = self.env.step(1)
        self.env.steps_remaining = self.env.max_steps
        
        self.episodes += 1
        return self.get_obs(obs) 

    def get_obs(self, obs):
        resized = resize(obs['image'], self.size, anti_aliasing=True)
        if self.grayscale:
            return np.dot(resized, [0.2989, 0.5870, 0.1140])
        else:
            return resized


class MinigridFeatureWrapper(Wrapper):
    
    def __init__(self, env, num_features=64, fixed_feature_file=None, terminate=False, reset_same=False, reset_episodes=1):
        super().__init__(env)
        self.env = env
        if fixed_feature_file is not None:
            self.feature_map = np.load(fixed_feature_file)
        else:
            self.feature_map = np.random.rand(19, 19, 4, num_features)

        self.terminate = terminate

        self.reset_same = reset_same
        self.start_pos = get_random_start()
        self.reset_episodes = reset_episodes
        self.episodes = 0

        self.observation_space = Box(0, 1, (self.feature_map.shape[3], ))
        self.action_space = Discrete(4)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up
        self.env.unwrapped.agent_dir = action
        _, reward, done, info = self.env.step(2)
        pos = tuple(self.env.unwrapped.agent_pos)
        obs = self.get_obs(pos, action)
        if not self.terminate:
            done = self.env.steps_remaining == 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()

        if self.reset_same or self.episodes != self.reset_episodes:
            self.env.unwrapped.agent_pos = self.start_pos
        else:
            self.start_pos = get_random_start()
            self.env.unwrapped.agent_pos = self.start_pos
            self.episodes = 0
        
        self.episodes += 1
        pos = self.env.unwrapped.agent_pos
        direction = self.env.unwrapped.agent_dir
        return self.get_obs(pos, direction)

    def get_obs(self, pos, direction):
        return self.feature_map[pos[0], pos[1], direction]


class MinigridActionFeatureWrapper(Wrapper):
    
    def __init__(self, env, num_features=64, fixed_feature_file=None, terminate=False, reset_same=False, reset_episodes=1):
        super().__init__(env)
        self.env = env
        if fixed_feature_file is not None:
            self.feature_map = np.load(fixed_feature_file)
        else:
            self.feature_map = np.random.rand(19, 19, 4, 4, num_features)

        self.terminate = terminate

        self.reset_same = reset_same
        self.start_pos = get_random_start()
        self.reset_episodes = reset_episodes
        self.episodes = 0

        self.observation_space = Box(0, 1, (4, self.feature_map.shape[4], ))
        self.action_space = Discrete(4)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up
        self.env.unwrapped.agent_dir = action
        _, reward, done, info = self.env.step(2)
        pos = tuple(self.env.unwrapped.agent_pos)
        obs = self.get_obs(pos, action)
        if not self.terminate:
            done = self.env.steps_remaining == 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()

        if self.reset_same or self.episodes != self.reset_episodes:
            self.env.unwrapped.agent_pos = self.start_pos
        else:
            self.start_pos = get_random_start()
            self.env.unwrapped.agent_pos = self.start_pos
            self.episodes = 0
        
        self.episodes += 1
        pos = self.env.unwrapped.agent_pos
        direction = self.env.unwrapped.agent_dir
        return self.get_obs(pos, direction)

    def get_obs(self, pos, direction):
        return self.feature_map[pos[0], pos[1], direction]


class MinigridTabularFeatureWrapper(Wrapper):
    
    def __init__(self, env, num_features=8, sigma=1, reset_same=False, reset_episodes=1):
        super().__init__(env)
        self.env = env
        self.one_hot = np.identity(361)
        self.feature_map = np.random.rand(19, 19, num_features)
        # self.feature_map = self.feature_map / self.feature_map.sum(axis=2, keepdims=True)

        self.reset_same = reset_same
        self.start_pos = get_random_start()
        self.reset_episodes = reset_episodes
        self.episodes = 0

        self.observation_space = Dict({"position": Box(0, 1, (361, )), "features": Box(0, 1, (num_features, ))})
        self.action_space = Discrete(4)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up
        self.env.unwrapped.agent_dir = action
        _, reward, done, info = self.env.step(2)
        pos = tuple(self.env.unwrapped.agent_pos)
        obs = self.get_obs(pos)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()

        if self.reset_same or self.episodes != self.reset_episodes:
            self.env.unwrapped.agent_pos = self.start_pos
        else:
            self.start_pos = get_random_start()
            self.env.unwrapped.agent_pos = self.start_pos
            self.episodes = 0
        
        self.episodes += 1
        pos = self.env.unwrapped.agent_pos
        return self.get_obs(pos)

    def get_obs(self, pos):
        return {
            "position": self.one_hot[pos[0] * 19 + pos[1]],
            "features": self.feature_map[tuple(pos)]
        }
        

class MinigridOneHotWrapper(Wrapper):
    
    def __init__(self, env, reset_same=False, reset_episodes=1):
        super().__init__(env)
        self.env = env
        self.one_hot = np.identity(361)

        self.reset_same = reset_same
        self.start_pos = get_random_start()
        self.reset_episodes = reset_episodes
        self.episodes = 0

        self.observation_space = Box(0, 1, (361, ))
        self.action_space = Discrete(4)

    def step(self, action):
        # 0 -- right, 1 -- down, 2 -- left, 3 -- up
        self.env.unwrapped.agent_dir = action
        _, reward, done, info = self.env.step(2)
        pos = tuple(self.env.unwrapped.agent_pos)
        obs = self.get_obs(pos)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()

        if self.reset_same or self.episodes != self.reset_episodes:
            self.env.unwrapped.agent_pos = self.start_pos
        else:
            self.start_pos = get_random_start()
            self.env.unwrapped.agent_pos = self.start_pos
            self.episodes = 0
        
        self.episodes += 1
        pos = tuple(self.env.unwrapped.agent_pos)
        return self.get_obs(pos)

    def get_obs(self, pos):
        return self.one_hot[pos[0] * 19 + pos[1]]


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


def make(*args, info_example=None, mode=None, minigrid_config=None, **kwargs):
    if minigrid_config is not None:
        num_features = minigrid_config.get('num_features', 64)
        fixed_feature_file = minigrid_config.get('fixed_feature_file', None)
        sigma = minigrid_config.get('sigma', 0.5)

        size = minigrid_config.get('size', (19, 19))
        grayscale = minigrid_config.get('grayscale', False)
        collection = minigrid_config.get('collection', False)

        terminate = minigrid_config.get('terminate', False)
        reset_same = minigrid_config.get('reset_same', False)
        reset_episodes = minigrid_config.get('reset_episodes', 1)
    
        max_steps = minigrid_config.get('max_steps', 500)

        if mode == 'multiroom':
            num_rooms = minigrid_config.get('num_rooms', 10)
            room_size = minigrid_config.get('room_size', 10)
            env = MultiRoomEnv(num_rooms, num_rooms, room_size)
            env.max_steps = max_steps
            env = RGBImgObsWrapper(ReseedWrapper(env))
            env = MinigridMultiRoomWrapper(env, num_rooms=num_rooms, size=size, grayscale=grayscale, terminate=terminate, reset_same=reset_same, reset_episodes=reset_episodes)
            
            oracle = minigrid_config.get('oracle', False)
            if oracle:
                epsilon = minigrid_config.get('epsilon', 0.05)
                env = MinigridMultiRoomOracleWrapper(env, epsilon=epsilon)
            return GymEnvWrapper(env)
        else:
            env = gym.make(*args, **kwargs)
            env.max_steps = max_steps
            env = ReseedWrapper(env)
            if mode == 'image':
                env = RGBImgObsWrapper(env)
                return GymEnvWrapper(MinigridImageWrapper(env, size=size, grayscale=grayscale, collection=collection, terminate=terminate, reset_same=reset_same, reset_episodes=reset_episodes))
            elif mode == 'one-hot':
                return GymEnvWrapper(MinigridOneHotWrapper(RGBImgObsWrapper(env), reset_same=reset_same, reset_episodes=reset_episodes))
            elif mode == 'one-hot-features':
                return GymEnvWrapper(MinigridTabularFeatureWrapper(RGBImgObsWrapper(env), num_features=num_features, sigma=sigma, reset_same=reset_same, reset_episodes=reset_episodes))
            elif mode == 'features':
                return GymEnvWrapper(MinigridFeatureWrapper(RGBImgObsWrapper(env), num_features=num_features, fixed_feature_file=fixed_feature_file, terminate=terminate, reset_same=reset_same, reset_episodes=reset_episodes))
            elif mode == 'action-features':
                return GymEnvWrapper(MinigridActionFeatureWrapper(RGBImgObsWrapper(env), num_features=num_features, fixed_feature_file=fixed_feature_file, terminate=terminate, reset_same=reset_same, reset_episodes=reset_episodes))
    elif info_example is None:
        return GymEnvWrapper(gym.make(*args, **kwargs))
    else:
        return GymEnvWrapper(EnvInfoWrapper(
            gym.make(*args, **kwargs), info_example))
