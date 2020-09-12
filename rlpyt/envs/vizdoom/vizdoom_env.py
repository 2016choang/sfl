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


EnvInfo = namedtuple("EnvInfo", ["traj_done", "position", "goal"])

W, H = (84, 84)

GOAL_DISTANCE_ALLOWANCE = 63

def check_if_close(first_point, second_point, threshold=GOAL_DISTANCE_ALLOWANCE):
    if ((first_point[0] - second_point[0]) ** 2 +
        (first_point[1] - second_point[1]) ** 2 <= threshold ** 2):
        return True
    else:
        return False

class VizDoomEnv(Env):

    def __init__(self,
                 config,
                 seed,
                 goal_position,
                 goal_close_terminate=False,
                 start_position=None,
                 grayscale=True,
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).
                 num_samples=100,
                 map_id=None,
                 full_action_set=True,
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
        if full_action_set:
            self._action_set = [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        else:
            self._action_set = [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        self._action_space = IntBox(low=0, high=len(self._action_set), dtype='long')
        if self.grayscale:
            obs_shape = (num_img_obs, H, W)
        else:
            if self.num_img_obs > 1:
                obs_shape = (self.num_img_obs, self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
            else:
                obs_shape = (self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        self._observation_space = IntBox(low=0, high=255, shape=obs_shape,
            dtype="uint8")
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")

        self.game.new_episode()
        self.set_start_state(start_position)
        self.set_goal_state(goal_position)

        state = self.game.get_state()
        sector_lines = np.array([[l.x1, l.x2, l.y1, l.y2] for s in state.sectors for l in s.lines if l.is_blocking])
        self.min_x = sector_lines[:, :2].min()
        self.max_x = sector_lines[:, :2].max()
        self.min_y = sector_lines[:, 2:].min()
        self.max_y = sector_lines[:, 2:].max()
        self.bin_size = 50
        x_len = int(self.max_x - self.min_x + self.bin_size) // self.bin_size
        y_len = int(self.max_y - self.min_y + self.bin_size) // self.bin_size
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
    
    def generate_samples(self):
        state = self.game.get_state()
        self.sample_states = [self.start_info, self.goal_info]
        self.sample_sectors = [(*self.start_info[1][:2], 0), (*self.goal_info[1][:2], 1)]
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
                sample_state, position = self.get_obs_at([x, y, 0])
                self.sample_states.append(sample_state)
                centroid = (sector_lines[:, :2].mean(), sector_lines[:, 2:].mean())
                self.sample_sectors.append((*centroid, i))
                self.sample_positions.append(position)

        self.sample_states = np.array(self.sample_states)
        self.sample_sectors = np.array(self.sample_sectors)
        self.sample_positions = np.array(self.sample_positions)

    def set_record_files(self, files):
        self.record_files = deque(files)

    def reset_logging(self):
        self.visited += self.visited_interval
        self.visited_interval[:] = 0

    def reset(self):
        if self.current_record_file:
            self.game.send_game_command('stop')
            self.game.close()
            self.game.init()
            time.sleep(5)
            self.current_record_file = None
        self._reset_obs()
        if self.record_files:
            self.current_record_file = self.record_files.popleft()
            self.game.new_episode(self.current_record_file)
        else:
            self.game.new_episode()
        if self.start_position is not None:
            self.get_obs_at(self.start_position)
        self.state = self.game.get_state()

        x, y = self.state.game_variables[:2]
        x = int(round(x - self.min_x)) // self.bin_size
        y = int(round(y - self.min_y)) // self.bin_size
        self.visited_interval[x, y] += 1

        new_obs = self.state.screen_buffer
        self._update_obs(new_obs)
        return self.get_obs()

    def step(self, action, position=None):
        if action == -1:
            if position is None:
                raise RuntimeError('No teleport position given!')
            reward = self.teleport(position)
        else:
            a = self._action_set[action]
            reward = self.game.make_action(a, self.frame_skip)
        done = self.game.is_episode_finished()

        if reward < 0.1:
            reward = 0

        reached_goal = False

        if not done:
            self.state = self.game.get_state()
            new_obs = self.state.screen_buffer
            x, y, theta = self.state.game_variables
            reached_goal = check_if_close(np.array([x, y]), self.goal_info[1][:2])
            if self.goal_close_terminate:
                done = reached_goal
            visited_x = int(round(x - self.min_x)) // self.bin_size
            visited_y = int(round(y - self.min_y)) // self.bin_size
            self.visited_interval[visited_x, visited_y] += 1
        else:
            # NOTE: when done, screen_buffer is invalid
            x, y, theta = 0, 0, 0
            if self.grayscale:
                new_obs = np.uint8(np.zeros(self._observation_space.shape[1:]))
            else:
                new_obs = np.uint8(np.zeros(self._observation_space.shape[1:]))

        if self.current_record_file: 
            if done:
                self.game.send_game_command('stop')
                self.game.close()
                self.game.init()
            else:
                if self.game.get_episode_time() + self.frame_skip >= self.game.get_episode_timeout():
                    self.game.send_game_command('stop')

        info = EnvInfo(traj_done=done, position=(x, y, theta), goal=reached_goal)

        self._update_obs(new_obs)
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10):
        img = self.get_obs()[-1]
        cv2.imshow('vizdoom', img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    def set_start_state(self, position):
        self._start_info = self.get_obs_at(position, idx=-1)
    
    def set_goal_state(self, position):
        self._goal_info = self.get_obs_at(position, idx=0)

    ###########################################################################
    # Helpers

    def _update_obs(self, new_obs):
        if self.grayscale:
            new_obs = np.transpose(new_obs, [1, 2, 0])
            img = cv2.resize(cv2.cvtColor(new_obs, cv2.COLOR_RGB2GRAY), (H, W), interpolation=cv2.INTER_LINEAR)
            # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
            self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])
        else:
            if self.num_img_obs > 1:
                self._obs = np.concatenate([self._obs[1:], new_obs[np.newaxis]])
            else:
                self._obs = new_obs

    def _reset_obs(self):
        self._obs[:] = 0
    
    def teleport(self, position):
        self.game.send_game_command('warp {} {}'.format(position[0], position[1]))
        cur_angle = self.game.get_game_variable(vzd.GameVariable.ANGLE)    
        turn_delta = int(cur_angle - position[2])
        self.game.make_action([0, 0, 0, 0, 0, 0, turn_delta], 1)
        reward = self.game.make_action([0, 0, 0, 0, 0, 0, 0], 1)
        return reward

    def get_obs_at(self, position=None, idx=0):
        state = self.game.get_state()

        if position is not None:
            self.teleport(position)

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
            new_obs = np.uint8(np.zeros(self._observation_space.shape))
            new_obs[-1] = img
            return new_obs, position
        else:
            if self.num_img_obs > 1:
                return_obs = np.uint8(np.zeros(self._observation_space.shape))
                return_obs[idx] = new_obs
                return return_obs, position
            else:
                return new_obs, position

    def plot_topdown(self, objects=True):
        if self.game.is_episode_finished():
            self.game.new_episode()
        state = self.game.get_state()
        
        if objects:
            plt.plot(*self.start_info[1][:2], color='red', marker='D')
            plt.plot(*self.goal_info[1][:2], color='green', marker='D')
        
        for s in state.sectors:
            # Plot sector on map
            for l in s.lines:
                if l.is_blocking:
                    plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)

    ###########################################################################
    # Properties

    @property
    def agent_pos(self):
        return self.state.game_variables
    
    @property
    def oracle_distance_matrix(self):
        return None
    
    @property
    def start_info(self):
        return self._start_info

    @property
    def goal_info(self):
        return self._goal_info
