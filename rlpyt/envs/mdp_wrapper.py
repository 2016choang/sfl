import scipy.ndimage
import tkinter
from PIL import Image
from PIL import ImageTk
import numpy as np
from PIL import Image
import scipy.ndimage
import networkx as nx
import numpy as np
import random
from gym import spaces

class GridWorld:
    def __init__(self, goal_locations, load_path=None, use_gui=False):
        self.action_space = spaces.Discrete(4)

        self.rewardFunction = None
        self.nb_actions = 4
        if load_path != None:
            self.read_file(load_path)
            self.set_goal_locations(goal_locations)

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.nb_rows, self.nb_cols, 3))
        self.agentX, self.agentY = self.startX, self.startY
        self.nb_states = self.nb_rows * self.nb_cols

        self.visited = np.zeros((self.nb_rows, self.nb_cols), dtype=int)

        self.use_gui = use_gui

        self.h = self.MDP.shape[0] * 42
        self.w = self.MDP.shape[1] * 42

        if self.use_gui:
            self.win = tkinter.Toplevel()

            screen_width = self.win.winfo_screenwidth()
            screen_height = self.win.winfo_screenheight()

            # calculate position x and y coordinates
            x = screen_width + 100
            y = screen_height + 100
            self.win.geometry('%sx%s+%s+%s' % (self.w, self.h, x, y))
            self.win.title("Gridworld")


    def get_current_state(self):
        return self.build_screen()

    def get_goal_state(self):
        goal_pos = np.array([self.goalX, self.goalY])
        orig_agentX, orig_agentY = self.agentX, self.agentY
        self.agentX, self.agentY = self.goalX, self.goalY

        screen = self.build_screen()
        self.agentX, self.agentY = orig_agentX, orig_agentY
        return screen, goal_pos

    def get_true_distances(self):
        h, w = self.nb_rows, self.nb_cols

        dist_matrix = np.zeros((h * w, h * w))
        valid = set()

        for i in range(h):
            for j in range(w):
                if self.MDP[i][j] != -1:
                    valid.add((i, j))

        for pos in valid:
            x, y = pos
            true_pos = y * w + x
            
            for adjacent in [[x-1, y], [x, y-1], [x+1, y], [x, y+1]]:
                adj_x, adj_y = adjacent
                if (adj_x, adj_y) in valid:
                    true_adj_pos = adj_y * w + adj_x
                    dist_matrix[true_pos, true_adj_pos] = 1

        G = nx.from_numpy_array(dist_matrix)
        lengths = nx.shortest_path_length(G)
        true_dist = np.zeros((w, h, w, h)) - 1

        for source, targets in lengths:
            source_x, source_y = source % w, source // w
            for target, dist in targets.items():
                target_x, target_y = target % w, target // w
                true_dist[source_x, source_y, target_x, target_y] = dist
        
        return true_dist

    def set_goal_locations(self, goal_locations):
        self.goal_locations = goal_locations

    def set_goal(self, episode_nb, goal_change):
        k = episode_nb // goal_change
        k = len(self.goal_locations) - 1 if k > len(self.goal_locations) - 1 else k
        goal_pair = self.goal_locations[k]
        self.goalX = goal_pair[0]
        self.goalY = goal_pair[1]

        return k

    def render(self, s):
        # time.sleep(0.1)
        # s = self.pix_state
        # screen = scipy.misc.imresize(s, [self.h, self.w,    3], interp='nearest')
        s = s.squeeze()
        screen = np.zeros((13, 13, 3))
        screen[s == 1] = np.array([255, 0, 0])
        screen[s == 2] = np.array([0, 255, 0])
        screen[s == 3] = np.array([0, 0, 255])
        screen = Image.fromarray(screen.astype('uint8')).resize((self.h, self.w))
        # screen = screen.resize((self.w, self.h))
        # screen_width = self.win.winfo_screenwidth()
        # screen_height = self.win.winfo_screenheight()
        # x = screen_width + 100
        # y = screen_height + 100
        #
        # self.win.geometry('%sx%s+%s+%s' % (512, 512, x, y))

        if self.use_gui:
            tkpi = ImageTk.PhotoImage(screen)
            label_img = tkinter.Label(self.win, image=tkpi)
            label_img.place(x=0, y=0,
                            width=self.w, height=self.h)

            # self.win.mainloop()  # wait until user clicks the window
            self.win.update_idletasks()
            self.win.update()

    def build_screen(self):
        mdp_screen = np.zeros_like(self.MDP)
        mdp_screen[self.MDP == -1] = 1
        mdp_screen[self.agentX, self.agentY] = 3
        mdp_screen[self.goalX, self.goalY] = 2
        self.pix_state = np.expand_dims(mdp_screen, 2)
        # self.pix_state /= 255.
        # mdp_screen[self.MDP == -1] = 255
        # self.pix_state = np.expand_dims(mdp_screen, 2)
        # self.pix_state = np.tile(self.pix_state, [1, 1, 3])
        # self.pix_state[self.agentX, self.agentY] = [0, 255, 0]
        # self.pix_state[self.goalX, self.goalY] = [255, 0, 0]
        #
        # self.pix_state /= 255.
        # self.pix_state -= 0.5
        # self.pix_state *= 2.
        # self.pix_state = scipy.misc.imresize(mdp_screen, [200, 200, 3], interp='nearest')
        return self.pix_state
        # return mdp_screen

    def reset(self):
        s = self.get_initial_state()
        screen = self.build_screen()
        stateIdx = self.get_state_index(self.agentX, self.agentY)
        self.visited[self.agentX, self.agentY] += 1

        return screen

    def read_file(self, load_path):
        with open(load_path, "r") as f:
            lines = f.readlines()
        self.nb_rows, self.nb_cols = lines[0].split(',')
        self.nb_rows, self.nb_cols = int(self.nb_rows), int(self.nb_cols)
        self.MDP = np.zeros((self.nb_rows, self.nb_cols))
        lines = lines[1:]
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                if lines[i][j] == '.':
                    self.MDP[i][j] = 0
                elif lines[i][j] == 'X':
                    self.MDP[i][j] = -1
                elif lines[i][j] == 'S':
                    self.MDP[i][j] = 0
                    self.startX = i
                    self.startY = j
                else:    # 'G'
                    self.MDP[i][j] = 0
                    self.goalX = i
                    self.goalY = j

    def get_state_index(self, x, y):
        idx = y + x * self.nb_cols
        return idx

    def get_start(self):
        while True:
            startX = random.randrange(0, self.nb_rows, 1)
            startY = random.randrange(0, self.nb_cols, 1)
            if self.MDP[startX][startY] != -1 and (startX != self.goalX or startY != self.goalY):
                break

        start_inx = self.get_state_index(startX, startY)

        return start_inx, startX, startY

    def get_initial_state(self):
        agent_state_index = self.get_state_index(self.startX, self.startY)
        # agent_state_index, self.startX, self.startY = self.get_start()
        self.agentX, self.agentY = self.startX, self.startY
        return agent_state_index

    def move_goal(self):
        while True:
            goalX = random.randrange(0, self.nb_rows, 1)
            goalY = random.randrange(0, self.nb_cols, 1)
            if self.MDP[goalX][goalY] != -1 and (goalX != self.startX or goalY != self.startY):
                break

        goal_indx = self.get_state_index(goalX, goalY)
        self.goalX = goalX
        self.goalY = goalY

    def get_next_state(self, a):
        action = ["up", "right", "down", "left", 'terminate']
        nextX, nextY = self.agentX, self.agentY

        try:
            if action[a] == 'terminate':
                return -1, -1
        except:
            print("ERROR")

        if self.MDP[self.agentX][self.agentY] != -1:
            if action[a] == 'up' and self.agentX > 0:
                nextX, nextY = self.agentX - 1, self.agentY
            elif action[a] == 'right' and self.agentY < self.nb_cols - 1:
                nextX, nextY = self.agentX, self.agentY + 1
            elif action[a] == 'down' and self.agentX < self.nb_rows - 1:
                nextX, nextY = self.agentX + 1, self.agentY
            elif action[a] == 'left' and self.agentY > 0:
                nextX, nextY = self.agentX, self.agentY - 1

        if self.MDP[nextX][nextY] != -1:
            return nextX, nextY
        else:
            return self.agentX, self.agentY

    def special_get_next_state(self, a, orig_nextX, orig_nextY):
        action = ["up", "right", "down", "left", 'terminate']

        nextX, nextY = orig_nextX, orig_nextY

        if action[a] == 'terminate':
            return -1, -1

        if self.MDP[orig_nextX][orig_nextY] != -1:
            if action[a] == 'up' and orig_nextY > 0:
                nextX, nextY = orig_nextX - 1, orig_nextY
            elif action[a] == 'right' and orig_nextY < self.nb_cols - 1:
                nextX, nextY = orig_nextX, orig_nextY + 1
            elif action[a] == 'down' and self.agentX < self.nb_rows - 1:
                nextX, nextY = orig_nextX + 1, orig_nextY
            elif action[a] == 'left' and orig_nextY > 0:
                nextX, nextY = orig_nextX, orig_nextY - 1

        if self.MDP[nextX][nextY] != -1:
            return nextX, nextY
        else:
            return orig_nextX, orig_nextY

    def is_terminal(self, nextX, nextY):
        if nextX == self.goalX and nextY == self.goalY:
            return True
        else:
            return False

    def get_next_reward(self, nextX, nextY):
        if self.rewardFunction is None:
            if nextX == self.goalX and nextY == self.goalY:
                reward = 1
            else:
                reward = 0
        elif len(self.rewardFunction) != self.nb_states and self.network != None and self.sess != None:
            currStateIdx = self.get_state_index(self.agentX, self.agentY)
            s, _, _ = self.get_state(currStateIdx)
            feed_dict = {self.network.observation: np.stack([s])}
            fi = self.sess.run(self.network.fi,
                                        feed_dict=feed_dict)[0]
            nextStateIdx = self.get_state_index(nextX, nextY)
            s1, _, _ = self.get_state(nextStateIdx)
            feed_dict = {self.network.observation: np.stack([s1])}
            fi1 = self.sess.run(self.network.fi,
                                                 feed_dict=feed_dict)[0]
            reward = self.cosine_similarity((fi1 - fi), self.rewardFunction)


        else:
            currStateIdx = self.get_state_index(self.agentX, self.agentY)
            nextStateIdx = self.get_state_index(nextX, nextY)

            reward = self.rewardFunction[nextStateIdx] \
                             - self.rewardFunction[currStateIdx]

        return reward

    def cosine_similarity(self, a, b):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        res = dot_product / ((norm_a + 1e-8) * (norm_b + 1e-8))
        # dot_product = np.sum(a*b)
        # norm_a = np.linalg.norm(np.asarray(a, np.float64))
        # norm_b = np.linalg.norm(np.asarray(b, np.float64))
        # res = dot_product / ((norm_a + + 1e-8) * (norm_b + + 1e-8))
        if np.isnan(res):
            print("NAN")
        return res

    def fake_get_state(self, idx):
        orig_agentX, orig_agentY = self.agentX, self.agentY
        x, y = self.get_state_xy(idx)
        self.agentX, self.agentY = x, y

        screen = self.build_screen()
        self.agentX, self.agentY = orig_agentX, orig_agentY

        return screen, x, y

    def get_state(self, idx):
        x, y = self.get_state_xy(idx)
        self.agentX, self.agentY = x, y

        screen = self.build_screen()

        return screen, x, y

    def not_wall(self, i, j):
        if self.MDP[i][j] != -1:
            return True
        else:
            return False

    def get_state_xy(self, idx):
        y = idx % self.nb_cols
        x = int((idx - y) / self.nb_cols)

        return x, y

    def get_next_state_and_reward(self, currState, a):
        if currState == self.nb_states:
            return currState, 0

        tmpx, tmpy = self.agentX, self.agentY
        self.agentX, self.agentY = self.get_state_xy(currState)
        nextX, nextY = self.agentX, self.agentY

        nextStateIdx = None
        reward = None

        nextX, nextY = self.get_next_state(a)
        if nextX != -1 and nextY != -1:    # If it is not the absorbing state:
            reward = self.get_next_reward(nextX, nextY)
            nextStateIdx = self.get_state_index(nextX, nextY)
        else:
            reward = 0
            nextStateIdx = self.nb_states

        self.agentX, self.agentY = tmpx, tmpy

        return nextStateIdx, reward

    def get_agent(self):
        return self.agentX, self.agentY

    def step(self, a):
        nextX, nextY = self.get_next_state(a)

        self.agentX, self.agentY = nextX, nextY
        self.visited[self.agentX, self.agentY] += 1

        done = False
        if self.is_terminal(nextX, nextY):
            done = True

        reward = self.get_next_reward(nextX, nextY)
        nextStateIdx = self.get_state_index(nextX, nextY)

        screen = self.build_screen()

        return screen, reward, done, nextStateIdx

    def fake_step(self, a):
        orig_agentX, orig_agentY = self.agentX, self.agentY
        nextX, nextY = self.get_next_state(a)

        self.agentX, self.agentY = nextX, nextY

        done = False
        if self.is_terminal(nextX, nextY):
            done = True

        reward = self.get_next_reward(nextX, nextY)
        nextStateIdx = self.get_state_index(nextX, nextY)

        screen = self.build_screen()

        self.agentX, self.agentY = orig_agentX, orig_agentY

        return screen, reward, done, nextStateIdx

    def special_step(self, a, last_state_idx):
        x, y = self.get_state_xy(last_state_idx)
        nextX, nextY = self.special_get_next_state(a, x, y)

        # new_x, new_y = nextX, nextY
        self.agentX, self.agentY = nextX, nextY

        done = False
        if self.is_terminal(nextX, nextY):
            done = True

        reward = self.get_next_reward(nextX, nextY)
        nextStateIdx = self.get_state_index(nextX, nextY)

        screen = self.build_screen()

        return screen, reward, done, nextStateIdx

    def get_action_set(self):
        return range(0, 4)

    def define_reward_function(self, vector):
        self.rewardFunction = vector

    def define_network(self, net):
        self.network = net

    def define_session(self, sess):
        self.sess = sess


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