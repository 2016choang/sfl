from collections import defaultdict
import copy
import itertools

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import floyd_warshall
from sklearn_extra.cluster import KMedoids
import torch
import torch.nn.functional as F

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.feature_dsr_agent import FeatureDSRAgent, IDFDSRAgent, TCFDSRAgent
from rlpyt.agents.dqn.dsr.landmarks import Landmarks
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

AgentInfo = namedarraytuple("AgentInfo", ["a", "mode"])


class LandmarkAgent(FeatureDSRAgent):

    def __init__(
            self,
            landmarks,
            landmark_update_interval=int(5e3),
            use_soft_q=False,
            use_oracle_graph=False,
            **kwargs):
        save__init__args(locals())
        self.explore = True
        self.landmarks = None
        self.landmark_mode_steps = 0
        self.reset_logging()
        super().__init__(**kwargs)
    
    def initialize(self, env_spaces, share_memory=False,
        global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        # Used for soft-Q action sampling
        self.soft_distribution = Categorical(dim=env_spaces.action.n)

    def reset_logging(self):
        # Percentage of times we reach the ith landmark
        self.landmark_attempts = np.zeros(self.max_landmarks)
        self.landmark_reaches = np.zeros(self.max_landmarks)
        self.landmark_true_reaches = np.zeros(self.max_landmarks)

        # End / start distance to ith landmark
        self.landmark_dist_completed = [[] for _ in range(self.max_landmarks)]

        # End / start distance to goal landmark
        self.goal_landmark_dist_completed = []

        if self.landmarks is not None:
            self.landmarks.reset_logging()

    @property
    def landmarks(self):
        if self._mode == 'eval':
            return self._eval_landmarks
        else:
            return self._landmarks

    @torch.no_grad()
    def initialize_landmarks(self, train_envs, eval_envs, goal, oracle_distance_matrix):
        # Create Landmarks object
        self._landmarks.initialize(train_envs)
        self._landmarks.oracle_distance_matrix = oracle_distance_matrix
        self.eval_envs = eval_envs 

        # Add goal observation as a landmark                           
        goal_obs, goal_pos = goal
        observation = torchify_buffer(goal_obs).unsqueeze(0).float()

        model_inputs = buffer_to(observation,
                device=self.device)
        features = self.feature_model(model_inputs, mode='encode')

        model_inputs = buffer_to(features,
                device=self.device)
        dsr = self.model(model_inputs, mode='dsr')

        self._landmarks.add_landmark(observation, features, dsr, goal_pos)

        # if self.use_oracle_landmarks:
        #     self.landmarks = self.oracle_landmarks
        #     self.max_landmarks = self.oracle_max_landmarks

    @torch.no_grad()
    def update_landmarks(self, itr):
        if self.landmarks is not None and itr % self.landmark_update_interval == 0:
            # Update features and successor features of existing landmarks
            observation = self.landmarks.observations
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.feature_model(model_inputs, mode='encode')
            self.landmarks.set_features(features)
            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')
            self.landmarks.set_dsr(dsr)

            self.landmarks.update()

            # Add new landmarks
            if self.landmarks.potential_landmarks:
                observation = self.landmarks.potential_landmarks['observation']
                unique_idxs = np.unique(observation.numpy(), return_index=True, axis=0)[1]
                position = self.landmarks.potential_landmarks['positions'][unique_idxs]

                observation = observation[unique_idxs]

                model_inputs = buffer_to(observation,
                    device=self.device)
                features = self.feature_model(model_inputs, mode='encode')
                self.landmarks.set_features(features)
                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')

                self.landmarks.add_landmarks(observation, features, dsr, position)

            self.explore = True

            # # Generate landmark graph and path between start and goal landmarks
            # self.generate_graph()

    def generate_graph(self):
        # Use true distance edges given by oracle
        if self.use_oracle_graph:
            self.landmarks.generate_true_graph(self.env_true_dist)

        # Use estimated edges
        else:
            self.landmarks.generate_graph(self._mode)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward, position=None):
        if self.landmarks is not None:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            # # Add landmarks if in exploration mode during training
            # if self.explore and self._mode != 'eval' and not self.use_oracle_landmarks:
            #     self.landmarks.add_landmark(observation.float(), features, dsr, position)

            self.landmarks.set_paths(dsr, position, self._mode)

            landmarks_dsr, landmark_mode = self.get_landmarks_data(self, dsr, self._mode)

        # Default exploration (uniform random) policy
        action = torch.randint_like(prev_action, high=self.distribution.dim)

        # Landmark subgoal policy (SF-based Q values)
        current_dsr = dsr[landmark_mode] / torch.norm(dsr[landmark_mode], p=2, dim=2, keepdim=True)
        q_values = torch.matmul(current_dsr, landmarks_dsr).cpu()
    
        if self.use_soft_q:
            # Select action based on softmax of Q as probabilities
            prob = F.softmax(q_values, dim=1)
            landmark_action = self.soft_distribution.sample(DistInfo(prob=prob))
        else:
            # Select action based on epsilon-greedy of Q
            landmark_action = self.distribution.sample(q_values)

        action[landmark_mode] = landmark_action

        # Try to enter landmark mode in training
        if self._mode != 'eval' and self.landmarks:
            self.landmarks.enter_landmark_mode()

        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)

    def reset(self):
        if self.landmarks:
            # Always start in landmarks mode
            self.landmarks.enter_landmark_mode(override=-1)

    def reset_one(self, idx):
        if self.landmarks:
            # Always start in landmarks mode
            self.landmarks.enter_landmark_mode(override=idx)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self._eval_landmarks = copy.deepcopy(self._landmarks)
        self._eval_landmarks.initialize(self.eval_envs, 'eval')

class LandmarkIDFAgent(LandmarkAgent, IDFDSRAgent):

    def __init__(self, **kwargs):
        LandmarkAgent.__init__(self, **kwargs)
        IDFDSRAgent.__init__(self)

class LandmarkTCFAgent(LandmarkAgent, TCFDSRAgent):

    def __init__(self, **kwargs):
        LandmarkAgent.__init__(self, **kwargs)
        TCFDSRAgent.__init__(self)
