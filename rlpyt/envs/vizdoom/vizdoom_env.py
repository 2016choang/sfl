from collections import namedtuple
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import vizdoom as vzd

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


EnvInfo = namedtuple("EnvInfo", ["traj_done"])

W, H = (84, 84)

class VizDoomEnv(Env):

    def __init__(self,
                 config,
                 seed,
                 goal_position,
                 goal_angle=0,
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).
                 ):
        save__init__args(locals())

        # init VizDoom game
        self.game = vzd.DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(False)
        self.game.set_seed(seed)
        self.game.init()

        # Spaces
        self._action_set = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]] 
        self._action_space = IntBox(low=0, high=len(self._action_set), dtype='long')
        obs_shape = (num_img_obs, H, W)
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

        self.sample_states = self.get_initial_landmarks()

        for s in state.sectors:
            sector_lines = np.array([[l.x1, l.x2, l.y1, l.y2] for l in s.lines])
            centroid = (int(sector_lines[:, :2].mean()), int(sector_lines[:, 2:].mean()))
            sample_state = self.get_obs_at(centroid)
            self.sample_states.append(sample_state)

    def reset_logging(self):
        self.visited += self.visited_interval
        self.visited_interval[:] = 0

    def reset(self):
        self._reset_obs()
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
        info = EnvInfo(traj_done=done)

        if reward < 0.1:
            reward = 0

        if not done:
            self.state = self.game.get_state()
            new_obs = self.state.screen_buffer
            x, y = self.state.game_variables[:2]
            x = int(round(x - self.min_x)) // 50
            y = int(round(y - self.min_y)) // 50
            self.visited_interval[x, y] += 1
        else:
            # NOTE: when done, screen_buffer is invalid
            new_obs = np.uint8(np.zeros(self._observation_space.shape[1:]))

        self._update_obs(new_obs)
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10):
        img = self.get_obs()[-1]
        cv2.imshow('vizdoom', img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    def get_initial_landmarks(self):
        return [(self.goal_state, self.goal_position), (self.start_state, self.start_position)]

    def set_start_state(self):
        self.start_state, self.start_position = self.get_cur_obs()
    
    def set_goal_state(self, position, angle=0):
        self.goal_state, self.goal_position = self.get_obs_at(position, angle)

    ###########################################################################
    # Helpers

    def _update_obs(self, new_obs):
        img = cv2.resize(new_obs, (W, H), cv2.INTER_LINEAR)
        # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
        self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])

    def _reset_obs(self):
        self._obs[:] = 0

    def get_cur_obs(self):
        state = self.game.get_state()
        new_obs = state.screen_buffer
        position = state.game_variables[:2]

        img = cv2.resize(new_obs, (W, H), cv2.INTER_LINEAR)
        return np.repeat(img[np.newaxis], self.num_img_obs, axis=0), position

    def get_obs_at(self, position, angle=0):
        state = self.game.get_state()
        cur_angle = self.game.get_game_variable(vzd.GameVariable.ANGLE)    
        turn_delta = int(cur_angle - angle)

        self.game.send_game_command('warp {} {}'.format(*position))
        self.game.make_action([0, 0, 0, 0, 0, turn_delta])
        state = self.game.get_state()

        new_obs = state.screen_buffer
        position = state.game_variables[:2]

        img = cv2.resize(new_obs, (W, H), cv2.INTER_LINEAR)
        return np.repeat(img[np.newaxis], self.num_img_obs, axis=0), position

    def plot_topdown(self):
        state = self.game.get_state()
        
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
