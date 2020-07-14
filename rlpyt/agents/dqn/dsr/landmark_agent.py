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
            landmarks=None,
            use_soft_q=False,
            use_oracle_graph=False,
            **kwargs):
        self._landmarks = landmarks
        local_args = locals()
        local_args.pop('landmarks')
        save__init__args(local_args)
        self.landmark_mode_steps = 0
        super().__init__(**kwargs)
    
    def initialize(self, env_spaces, share_memory=False,
        global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        # Used for soft-Q action sampling
        self.soft_distribution = Categorical(dim=env_spaces.action.n)

    def reset_logging(self):
        if self.landmarks:
            self.landmarks.reset_logging()

    @property
    def landmarks(self):
        if self._mode == 'eval':
            return self._eval_landmarks
        else:
            return self._landmarks
    
    @property
    def train_landmarks(self):
        return self._landmarks
    
    @property
    def eval_landmarks(self):
        return self._eval_landmarks

    @torch.no_grad()
    def initialize_landmarks(self, train_envs, eval_envs, initial_landmarks, oracle_distance_matrix):
        # Create Landmarks object
        self._landmarks.initialize(train_envs)
        self._landmarks.oracle_distance_matrix = oracle_distance_matrix
        self.eval_envs = eval_envs 

        # Add initial landmarks
        for obs, pos in initial_landmarks: 
            observation = torchify_buffer(obs).unsqueeze(0).float()

            model_inputs = buffer_to(observation,
                    device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                    device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            self._landmarks.force_add_landmark(observation, features, dsr, pos)

        self._landmarks.generate_graph()

    @torch.no_grad()
    def update_landmarks(self, itr):
        if self.landmarks:
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
                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')

                self.landmarks.add_landmarks(observation, features, dsr, position)

            # Reset landmark mode for all environments
            self.reset()

            # Generate landmark graph
            self.landmarks.generate_graph()

    @torch.no_grad()
    def get_norms(self):
        start_features_norm = torch.norm(self.landmarks.features[1], p=2).item()
        start_s_features_norm = torch.norm(self.landmarks.dsr[1], p=2).item()
        goal_features_norm = torch.norm(self.landmarks.features[0], p=2).item()
        goal_s_features_norm = torch.norm(self.landmarks.dsr[0], p=2).item()
        return start_features_norm, start_s_features_norm, goal_features_norm, goal_s_features_norm

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward, position=None):
        # Default exploration (uniform random) policy
        action = torch.randint_like(prev_action, high=self.distribution.dim)
        mode = torch.zeros_like(prev_action, dtype=bool)

        # Use landmark policy sometimes
        if self.landmarks:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            observation = observation.float()

            # Add potential landmarks during training
            if self._mode != 'eval':
                self.landmarks.add_potential_landmark(observation, dsr, position)

            self.landmarks.set_paths(dsr, position)

            landmarks_dsr, landmark_mode = self.landmarks.get_landmarks_data(observation, dsr, position)

            if np.any(landmark_mode):
                # Landmark subgoal policy (SF-based Q values)
                current_dsr = dsr[landmark_mode] / torch.norm(dsr[landmark_mode], p=2, dim=2, keepdim=True)
                q_values = torch.sum(current_dsr * landmarks_dsr.unsqueeze(1), dim=2).cpu()

                if self.use_soft_q:
                    # Select action based on softmax of Q as probabilities
                    prob = F.softmax(q_values, dim=1)
                    landmark_action = self.soft_distribution.sample(DistInfo(prob=prob))
                else:
                    # Select action based on epsilon-greedy of Q
                    landmark_action = self.distribution.sample(q_values)

                action[landmark_mode] = landmark_action

            # Try to enter landmark mode in training
            if self._mode != 'eval':
                self.landmarks.enter_landmark_mode()
            
            mode = torch.from_numpy(landmark_mode)

        agent_info = AgentInfo(a=action, mode=mode)
        return AgentStep(action=action, agent_info=agent_info)

    def reset(self):
        if self.landmarks:
            # Always start in landmarks mode
            self.landmarks.enter_landmark_mode(override=-1)

    def reset_one(self, idx):
        if self.landmarks:
            # Always start in landmarks mode
            self.landmarks.enter_landmark_mode(override=idx)
    
    def get_landmark_mode(self, idx):
        if self.landmarks:
            return self.landmarks.landmark_mode[idx]
        else:
            return False

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self._eval_landmarks = copy.deepcopy(self._landmarks)
        self._eval_landmarks.initialize(self.eval_envs, 'eval')
        self._eval_landmarks.generate_graph()
    
    def log_eval(self, idx, pos):
        self.landmarks.log_eval(idx, pos)

class LandmarkIDFAgent(LandmarkAgent, IDFDSRAgent):

    def __init__(self, **kwargs):
        LandmarkAgent.__init__(self, **kwargs)
        IDFDSRAgent.__init__(self)

class LandmarkTCFAgent(LandmarkAgent, TCFDSRAgent):

    def __init__(self, **kwargs):
        LandmarkAgent.__init__(self, **kwargs)
        TCFDSRAgent.__init__(self)
