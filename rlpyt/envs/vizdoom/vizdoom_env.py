from collections import deque, namedtuple
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import vizdoom as vzd

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


EnvInfo = namedtuple("EnvInfo", ["traj_done", "position"])

W, H = (84, 84)

class VizDoomEnv(Env):

    def __init__(self,
                 config,
                 seed,
                 goal_position,
                 goal_angle=0,
                 grayscale=True,
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).
                 num_samples=100,
                 map_id=None,
                 ):
        save__init__args(locals())

        # init VizDoom game
        self.game = vzd.DoomGame()
        self.game.load_config(config)
        if map_id:
            self.game.set_doom_map(map_id)
        self.game.set_seed(seed)
        self.game.init()

        # Spaces
        self._action_set = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ]
        self._action_space = IntBox(low=0, high=len(self._action_set), dtype='long')
        if self.grayscale:
            obs_shape = (num_img_obs, H, W)
        else:
            obs_shape = (self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        self._observation_space = IntBox(low=0, high=255, shape=obs_shape,
            dtype="uint8")
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")

        self.game.new_episode()
        self.set_start_state()
        self.set_goal_state(goal_position, goal_angle)

        state = self.game.get_state()
        sector_lines = np.array([[l.x1, l.x2, l.y1, l.y2] for s in state.sectors for l in s.lines if l.is_blocking])
        self.min_x = sector_lines[:, :2].min()
        self.max_x = sector_lines[:, :2].max()
        self.min_y = sector_lines[:, 2:].min()
        self.max_y = sector_lines[:, 2:].max()
        x_len = int(self.max_x - self.min_x + 50) // 50
        y_len = int(self.max_y - self.min_y + 50) // 50
        self.visited = np.zeros((x_len, y_len), dtype=int)
        self.visited_interval = np.zeros((x_len, y_len), dtype=int)

        self.record_files = None
        self.current_record_file = None

        if self.num_samples == -1:
            self.generate_full_episode()
        else:
            self.generate_samples()

    def generate_full_episode(self):
        self.reset()
        self.sample_states = []
        self.sample_sectors = []
        self.sample_positions = []
        while True:
            state, reward, done, info = self.step(self.action_space.sample())
            if done:
                break
            self.sample_states.append(state)
            self.sample_positions.append(info.position)
        self.sample_states = np.array(self.sample_states)[::4]
        self.sample_positions = np.array(self.sample_positions)[::4]
        self.sample_positions[:, 0] *= self.max_x
        self.sample_positions[:, 1] *= self.max_y
    
    def generate_samples(self):
        state = self.game.get_state()
        self.sample_states = [self.start_info, self.goal_info]
        self.sample_sectors = [(*self.start_info[1], 0), (*self.goal_info[1], 1)]
        self.sample_positions = [self.start_info[1], self.goal_info[1]]

        samples_per_sector = self.num_samples // len(state.sectors)

        for i, s in enumerate(state.sectors, 2):
            sector_lines = np.array([[l.x1, l.x2, l.y1, l.y2] for l in s.lines])
            min_x = sector_lines[:, :2].min()
            max_x = sector_lines[:, :2].max()
            min_y = sector_lines[:, 2:].min()
            max_y = sector_lines[:, 2:].max()

            sample_x = np.random.uniform(min_x, max_x, samples_per_sector)
            sample_y = np.random.uniform(min_y, max_y, samples_per_sector)
            for x, y in zip(sample_x, sample_y):
                sample_state, position = self.get_obs_at([x, y], self.goal_angle)
                self.sample_states.append(sample_state)
                centroid = (sector_lines[:, :2].mean(), sector_lines[:, 2:].mean())
                self.sample_sectors.append((*centroid, i))
                self.sample_positions.append(position)

        self.sample_states = np.array(self.sample_states)
        self.sample_sectors = np.array(self.sample_sectors)
        self.sample_positions = np.array(self.sample_positions)
        self.sample_positions[:, 0] *= self.max_x
        self.sample_positions[:, 1] *= self.max_y

    def set_record_files(self, files):
        self.record_files = deque(files)

    def reset_logging(self):
        self.visited += self.visited_interval
        self.visited_interval[:] = 0

    def reset(self):
        self._reset_obs()
        if self.record_files:
            self.current_record_file = self.record_files.popleft()
            self.game.new_episode(self.current_record_file)
        else:
            self.game.new_episode()
        self.state = self.game.get_state()

        x, y = self.state.game_variables[:2]
        x = int(round(x - self.min_x)) // 50
        y = int(round(y - self.min_y)) // 50
        self.visited_interval[x, y] += 1

        new_obs = self.state.screen_buffer
        self._update_obs(new_obs)
        return self.get_obs()

    def step(self, action):
        a = self._action_set[action]
        reward = self.game.make_action(a, self.frame_skip)
        done = self.game.is_episode_finished()

        if reward < 0.1:
            reward = 0

        if not done:
            self.state = self.game.get_state()
            new_obs = self.state.screen_buffer
            x, y, theta = self.state.game_variables
            visited_x = int(round(x - self.min_x)) // 50
            visited_y = int(round(y - self.min_y)) // 50
            self.visited_interval[visited_x, visited_y] += 1
        else:
            if self.current_record_file:
                self.game.close()
                self.game.init()
                self.current_record_file = None
            # NOTE: when done, screen_buffer is invalid
            x, y, theta = 0, 0, 0
            if self.grayscale:
                new_obs = np.uint8(np.zeros(self._observation_space.shape[1:]))
            else:
                new_obs = np.uint8(np.zeros(self._observation_space.shape))

        info = EnvInfo(traj_done=done, position=(x / self.max_x, y / self.max_y, theta))

        self._update_obs(new_obs)
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10):
        img = self.get_obs()[-1]
        cv2.imshow('vizdoom', img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    def set_start_state(self):
        self.start_state, self.start_position = self.get_obs_at(full=False)
    
    def set_goal_state(self, position, angle=0):
        self.goal_state, self.goal_position = self.get_obs_at(position, angle, full=True)

    ###########################################################################
    # Helpers

    def _update_obs(self, new_obs):
        if self.grayscale:
            new_obs = np.transpose(new_obs, [1, 2, 0])
            img = cv2.resize(cv2.cvtColor(new_obs, cv2.COLOR_RGB2GRAY), (H, W), interpolation=cv2.INTER_LINEAR)
            # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
            self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])
        else:
            self._obs = new_obs

    def _reset_obs(self):
        self._obs[:] = 0

    def get_obs_at(self, position=None, angle=None, full=False):
        state = self.game.get_state()

        if position is not None:
            self.game.send_game_command('warp {} {}'.format(*position))

        turn_delta = 0
        if angle is not None:
            cur_angle = self.game.get_game_variable(vzd.GameVariable.ANGLE)    
            turn_delta = int(cur_angle - angle)

        if position is not None or angle is not None:
            self.game.make_action([0, 0, 0, 0, 0, 0, turn_delta])
            state = self.game.get_state()

        if self.game.is_episode_finished():
            position = np.array([0, 0, 0])
            if self.grayscale:
                new_obs = np.uint8(np.zeros(self._observation_space.shape[1:]))
            else:
                new_obs = np.uint8(np.zeros(self._observation_space.shape))
            self.game.new_episode()
        else:
            new_obs = state.screen_buffer
            position = state.game_variables

        if self.grayscale:
            new_obs = np.transpose(new_obs, [1, 2, 0])
            img = cv2.resize(cv2.cvtColor(new_obs, cv2.COLOR_RGB2GRAY), (H, W), interpolation=cv2.INTER_LINEAR)
            if full:
                return np.repeat(img[np.newaxis], self.num_img_obs, axis=0), position
            else:
                new_obs = np.uint8(np.zeros(self._observation_space.shape))
                new_obs[-1] = img
                return new_obs, position
        else:
            return new_obs, position

    def plot_topdown(self, objects=True):
        if self.game.is_episode_finished():
            self.game.new_episode()
        state = self.game.get_state()
        
        if objects:
            for o in state.objects:
                # Plot object on map
                if o.name == "DoomPlayer":
                    plt.plot(o.position_x, o.position_y, color='red', marker='D')
                else:
                    plt.plot(o.position_x, o.position_y, color='green', marker='D')
        
        for s in state.sectors:
            # Plot sector on map
            for l in s.lines:
                if l.is_blocking:
                    plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)

    ###########################################################################
    # Properties

    @property
    def agent_pos(self):
        return self.state.game_variables[:2]
    
    @property
    def oracle_distance_matrix(self):
        return None
    
    @property
    def start_info(self):
        return (self.start_state, self.start_position)

    @property
    def goal_info(self):
        return (self.goal_state, self.goal_position)
