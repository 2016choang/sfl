from collections import defaultdict
import copy
import itertools

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import floyd_warshall
from sklearn.manifold import TSNE
# from sklearn_extra.cluster import KMedoids
import torch
import torch.nn.functional as F

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.feature_dsr_agent import FeatureDSRAgent, IDFDSRAgent, TCFDSRAgent
from rlpyt.agents.dqn.dsr.landmarks import Landmarks
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

AgentInfo = namedarraytuple("AgentInfo", ["a", "mode", "subgoal"])


class LandmarkAgent(FeatureDSRAgent):

    def __init__(
            self,
            landmarks=None,
            use_soft_q=False,
            GT_subgoal_policy=False,
            **kwargs):
        self._landmarks = landmarks
        local_args = locals()
        local_args.pop('landmarks')
        save__init__args(local_args)
        self.landmark_mode_steps = 0
        self.reached_goal = False
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
    def initialize_landmarks(self, train_envs, eval_envs, lines):
        # Create Landmarks object
        self._landmarks.initialize(train_envs, self.model.feature_size, self.device)
        self._landmarks.lines = lines
        self.eval_envs = eval_envs 
    
    @torch.no_grad()
    def update_landmark_representation(self, itr):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            # Update features and successor features of existing landmarks
            # observation = self.landmarks.observations
            # model_inputs = buffer_to(observation,
            #     device=self.device)
            # features = self.feature_model(model_inputs, mode='encode')
            # self.landmarks.set_features(features)
            idxs = np.arange(self.landmarks.num_landmarks)
            for chunk_idxs in np.array_split(idxs, np.ceil(self.landmarks.num_landmarks / 128)):
                original_dsr.shape = self.landmarks.norm_dsr.shape
                features = self.landmarks.features[chunk_idxs] 
                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')
                dsr = dsr.reshape(original_dsr.shape)
                self.landmarks.set_dsr(dsr, chunk_idxs)

    @torch.no_grad()
    def update_landmark_graph(self, itr):
        if self.landmarks:
            self.landmarks.current_num_landmarks = self.landmarks.num_landmarks
            if self.landmarks.num_landmarks > 0:
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
        if len(prev_action.size()) == 0:
            subgoal = np.zeros((3, ))
        else:
            subgoal = np.zeros((len(observation), 3))

        # Use landmark policy sometimes
        if self.landmarks and not (self._mode == 'eval' and not np.any(self.landmarks.landmark_mode)):
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            observation = observation.float()

            # Add potential landmarks during training
            self.landmarks.analyze_current_state(features, dsr, position)

            self.landmarks.set_paths(position)

            landmarks_dsr, landmark_mode, subgoal_landmarks = self.landmarks.get_landmarks_data(position)

            if np.any(landmark_mode):
                if self.GT_subgoal_policy and self._mode != 'eval':
                    action[landmark_mode] = -1
                    subgoal[landmark_mode] = subgoal_landmarks
                else:
                    # Landmark subgoal policy (SF-based Q values)
                    current_dsr = dsr[landmark_mode] / torch.norm(dsr[landmark_mode], p=2, dim=2, keepdim=True) # |env| x A x 512
                    # goal SF |env| x 512 (SF of goal state averaged across actions)
                    q_values = torch.sum(current_dsr * landmarks_dsr.unsqueeze(1), dim=2).cpu() # <current SF, goal SF> = |env| x A

                    if self.use_soft_q:
                        # Select action based on softmax of Q as probabilities
                        prob = F.softmax(q_values, dim=1)
                        landmark_action = self.soft_distribution.sample(DistInfo(prob=prob))
                    else:
                        # Select action based on epsilon-greedy of Q
                        landmark_action = self.distribution.sample(q_values)

                    action[landmark_mode] = landmark_action
                
                self.landmarks.transition_subgoal_steps[landmark_mode] += 1

            self.landmarks.transition_random_steps[~landmark_mode] += 1

            # Try to enter landmark mode in training
            if self._mode != 'eval' and self.landmarks.current_num_landmarks > 0:
                self.landmarks.enter_landmark_mode()
            
            mode = torch.from_numpy(landmark_mode)

        agent_info = AgentInfo(a=action, mode=mode, subgoal=subgoal)
        return AgentStep(action=action, agent_info=agent_info)

    def reset(self, reset_landmarks=True):
        if self.landmarks and self.landmarks.current_num_landmarks > 0:
            # Always start in landmarks mode
            self.landmarks.enter_landmark_mode(override=-1)
            if reset_landmarks:
                self.landmarks.reset()

    def reset_one(self, idx):
        if self.landmarks and self.landmarks.current_num_landmarks > 0:
            # Always start in landmarks mode
            self.landmarks.enter_landmark_mode(override=idx)
            self.landmarks.reset(idx)
    
    def get_landmark_mode(self, idx):
        if self.landmarks:
            return self.landmarks.landmark_mode[idx]
        else:
            return False

    def eval_mode(self, itr, goal_info):
        super().eval_mode(itr)
        self._eval_landmarks = copy.deepcopy(self._landmarks)
        self._eval_landmarks.initialize(self.eval_envs, self.model.feature_size, self.device, 'eval')
        if self._eval_landmarks:
            obs, pos = goal_info

            observation = torchify_buffer(obs).unsqueeze(0).float()

            model_inputs = buffer_to(observation,
                    device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                    device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            self._eval_landmarks.force_add_landmark(features, dsr, pos)
            self._eval_landmarks.connect_goal()
    
    def log_eval(self, idx, pos):
        if self._eval_landmarks:
            self.landmarks.log_eval(idx, pos)

class LandmarkIDFAgent(LandmarkAgent, IDFDSRAgent):
    pass

class LandmarkVizDoomAgent(LandmarkAgent):

    @torch.no_grad()
    def get_representations(self, env):
        # Get features and SFs of sample states
        features = []
        s_features = []
        target_s_features = []

        for state in env.sample_states:
            model_inputs = buffer_to(torch.Tensor(state).unsqueeze(0),
                    device=self.device)            
            current_features = self.feature_model.get_features(model_inputs)
            features.append(current_features)

            model_inputs = buffer_to(current_features,
                device=self.device)
            current_s_features = self.model(model_inputs, mode='dsr')
            s_features.append(current_s_features)

            current_target_s_features = self.target_model(model_inputs, mode='dsr')
            target_s_features.append(current_target_s_features)

        return torch.stack(features), torch.stack(s_features), torch.stack(target_s_features), env.sample_positions

    @torch.no_grad()
    def get_representation_similarity(self, representation, mean_axes=None, subgoal_index=0):
        representation = representation.cpu().detach().numpy()
        if mean_axes:
            representation_matrix = representation.mean(axis=mean_axes)
        else:
            representation_matrix = representation
        representation_matrix = representation_matrix / np.linalg.norm(representation_matrix, ord=2, axis=-1, keepdims=True)

        subgoal_representation = representation_matrix[subgoal_index]
        similarity = np.matmul(representation_matrix, subgoal_representation)
        return similarity
    
    @torch.no_grad()
    def get_tsne(self, representation, mean_axes=None):
        representation = representation.cpu().detach().numpy()
        if mean_axes:
            representation_matrix = representation.mean(axis=mean_axes)
        else:
            representation_matrix = representation
        representation_matrix = representation_matrix / np.linalg.norm(representation_matrix, ord=2, axis=-1, keepdims=True)

        embeddings = TSNE(n_components=2).fit_transform(representation_matrix)
        return embeddings

    @torch.no_grad()
    def get_q_values(self, representation, mean_axes=None, subgoal_index=0):
        representation = representation.cpu().detach().numpy()
        if mean_axes:
            representation_matrix = representation.mean(axis=mean_axes)
        else:
            representation_matrix = representation
        representation_matrix = representation_matrix / np.linalg.norm(representation_matrix, ord=2, axis=-1, keepdims=True)

        subgoal_representation = representation_matrix[subgoal_index].mean(axis=0)
        q_values = np.matmul(representation_matrix, subgoal_representation)
        return q_values
