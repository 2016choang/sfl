from collections import defaultdict
import copy
import itertools

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.special import softmax
from sklearn_extra.cluster import KMedoids
import torch
import torch.nn.functional as F

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.idf_dsr_agent import IDFDSRAgent, AgentInfo
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

def get_true_pos(obs):
    h, w = obs.shape[:2]
    idx = np.argmax(obs[:, :, 0] - obs[:, :, 2])
    return [idx % w, idx // w]  

class Landmarks(object):

    def __init__(self, max_landmarks, threshold=0.75, landmark_paths=None):
        save__init__args(locals())
        self.num_landmarks = 0
        self.observations = None 
        self.features = None
        self.norm_features = None
        self.dsr = None
        self.norm_dsr = None
        self.visitations = None
        self.predecessors = None
        self.graph = None

        self.landmark_adds = 0
        self.landmark_removes = 0
    
    def reset_logging(self):
        self.landmark_adds = 0
        self.landmark_removes = 0

    def force_add_landmark(self, observation, features, dsr):
        if self.num_landmarks == 0:
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.visitations = np.array([0])
            self.num_landmarks += 1
        else:
            self.observations = torch.cat((self.observations, observation), dim=0)
            self.set_features(features, self.num_landmarks)
            self.set_dsr(dsr, self.num_landmarks)
            self.num_landmarks += 1
            self.visitations = np.append(self.visitations, 0)

    def force_remove_landmark(self):
        save_idx = range(self.num_landmarks - 1)

        self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]
        
        self.visitations = self.visitations[save_idx]

        self.num_landmarks -= 1

    def add_landmark(self, observation, features, dsr):
        if self.num_landmarks == 0:
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.visitations = np.array([0])
            self.num_landmarks += 1

            self.landmark_adds += 1
        else:
            norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T)  # cosine similarity of dsr

            if sum(similarity < self.threshold) >= self.num_landmarks - 1:
                self.landmark_adds += 1

                if self.num_landmarks < self.max_landmarks:
                    self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks)
                    self.num_landmarks += 1
                    self.visitations = np.append(self.visitations, 0)

                else:
                    landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
                    landmark_similarities[range(self.num_landmarks), range(self.num_landmarks)] = -2
                    idx = landmark_similarities.argmax().item()
                    a, b = idx // self.num_landmarks, idx % self.num_landmarks
                    if similarity[a] > similarity[b]:
                        replace_idx = a
                    else:
                        replace_idx = b
                    self.observations[replace_idx] = observation
                    self.set_features(features, replace_idx)
                    self.set_dsr(dsr, replace_idx)
                    self.visitations[replace_idx] = 0

    def set_features(self, features, idx=None):
        norm_features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        if self.features is None or idx is None:
            self.features = features
            self.norm_features = norm_features
        elif 0 <= idx and idx < self.num_landmarks:
            self.features[idx] = features
            self.norm_features[idx] = norm_features
        else:
            self.features = torch.cat((self.features, features), dim=0)
            self.norm_features = torch.cat((self.norm_features, norm_features), dim=0)

    def set_dsr(self, dsr, idx=None):
        dsr = dsr.mean(dim=1)
        norm_dsr = dsr / torch.norm(dsr, p=2, dim=1, keepdim=True)
        if self.dsr is None or idx is None:
            self.dsr = dsr
            self.norm_dsr = norm_dsr
        elif 0 <= idx and idx < self.num_landmarks:
            self.dsr[idx] = dsr
            self.norm_dsr[idx] = norm_dsr
        else:
            self.dsr = torch.cat((self.dsr, dsr), dim=0)
            self.norm_dsr = torch.cat((self.norm_dsr, norm_dsr), dim=0)

    def get_pos(self):
        observations = self.observations.detach().cpu().numpy()
        pos = np.zeros((len(observations), 2), dtype=int)

        for i, obs in enumerate(observations):
            pos[i] = get_true_pos(obs)
        return pos

    def generate_true_graph(self, env_true_dist, edge_threshold=None):
        # TODO: Hack to generate landmark graph based on true distances
        n_landmarks = len(self.norm_dsr)
        landmark_distances = np.zeros((n_landmarks, n_landmarks))

        landmark_pos = self.get_pos()
        for s in range(n_landmarks):
            s_x, s_y = landmark_pos[s]
            for t in range(n_landmarks):
                t_x, t_y = landmark_pos[t]
                if s_x == t_x and s_y == t_y:
                    landmark_distances[s, t] = 1
                else:
                    landmark_distances[s, t] = env_true_dist[s_x, s_y, t_x, t_y]

        try_distances = landmark_distances.copy()
        if edge_threshold is not None:
            non_edges = try_distances > edge_threshold
            try_distances[try_distances > edge_threshold] = 0

        G = nx.from_numpy_array(try_distances)
        if not nx.is_connected(G):
            avail = {index: x for index, x in np.ndenumerate(landmark_distances) if non_edges[index]}
            for edge in nx.k_edge_augmentation(G, 1, avail):
                try_distances[edge] = landmark_distances[edge]
            G = nx.from_numpy_array(try_distances)
            # Should be connected now!

        self.graph = G
        return self.graph
        
    def generate_graph(self):
        n_landmarks = len(self.norm_dsr)
        similarities = torch.clamp(torch.matmul(self.norm_dsr, self.norm_dsr.T), min=-1.0, max=1.0)
        similarities = similarities.detach().cpu().numpy()
        similarities[range(n_landmarks), range(n_landmarks)] = -2
        max_idx = similarities.argmax(axis=1)

        landmark_distances = 1.001 - similarities
        non_edges = similarities < self.threshold
        non_edges[range(n_landmarks), max_idx] = False
        landmark_distances[non_edges] = 0

        G = nx.from_numpy_array(landmark_distances)
        
        if not nx.is_connected(G):
            avail = {index: 1.001 - x for index, x in np.ndenumerate(similarities) if non_edges[index]}
            for edge in nx.k_edge_augmentation(G, 1, avail):
                landmark_distances[edge] = 1.001 - similarities[edge]
            G = nx.from_numpy_array(landmark_distances)
            # Should be connected now!

        self.graph = G
        return self.graph

    def generate_path(self, source, target):
        if self.landmark_paths is not None:
            paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, source, target, weight='weight'), self.landmark_paths))
            path_lengths = np.array([len(path) for path in paths])
            path_choice = np.random.choice(list(range(len(paths))), p=softmax(-1 * path_lengths))
            self.path = paths[path_choice]
        else:
            max_length = 0
            for path in nx.all_shortest_paths(self.graph, source, target, weight='weight'):
                if len(path) > max_length:
                    self.path = path
                    max_length = len(path)
        return self.path

    def prune_landmarks(self):
        landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
        save_idx = torch.sum(landmark_similarities < self.threshold, axis=1) >= (self.num_landmarks - 1)

        self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]
        
        save_idx = save_idx.detach().cpu().numpy()
        self.visitations = self.visitations[save_idx]

        self.landmark_removes += (self.num_landmarks - sum(save_idx))

        self.num_landmarks = sum(save_idx)


class LandmarkAgent(IDFDSRAgent):

    def __init__(
            self,
            landmark_mode_interval=100,
            landmark_update_interval=int(5e3),
            add_threshold=0.75,
            reach_threshold=0.95,
            max_landmarks=8,
            steps_per_landmark=25,
            edge_threshold=25,
            landmark_paths=None, 
            use_sf=False,
            use_soft_q=False,
            true_distance=False,
            steps_for_true_reach=2,
            oracle=False,
            use_oracle_landmarks=False,
            use_oracle_eval_landmarks=False,
            **kwargs):
        save__init__args(locals())
        self.landmark_mode_steps = 0
        self.explore = True
        self.landmarks = None
        self.update = False
        self.reset_logging()
        super().__init__(**kwargs)
    
    def initialize(self, env_spaces, share_memory=False,
        global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.soft_distribution = Categorical(dim=env_spaces.action.n)

    def reset_logging(self):
        self.correct_start_landmark = 0
        self.num_paths = 0

        self.dist_ratio_start_landmark = []

        self.path_freq = np.zeros(self.max_landmarks + 1)
        self.path_progress = np.zeros(self.max_landmarks + 1)
        self.true_path_progress = np.zeros(self.max_landmarks + 1)
 
        self.end_start_dist_progress = [[] for _ in range(self.max_landmarks + 1)]
        self.end_start_dist_ratio = []

        if self.landmarks is not None:
            self.landmarks.reset_logging()

    def set_oracle_landmarks(self, env):
        landmarks = env.get_oracle_landmarks()
        self.oracle_landmarks = Landmarks(len(landmarks), self.add_threshold, self.landmark_paths)
        for landmark in landmarks:
            observation = torchify_buffer(landmark).unsqueeze(0).float()

            model_inputs = buffer_to(observation,
                    device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                    device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            self.oracle_landmarks.force_add_landmark(observation, features, dsr)
        
        self.max_landmarks = len(landmarks)
        self.reset_logging()

    def set_env_true_dist(self, env):
        self.env_true_dist = env.get_true_distances()

    def update_landmarks(self, itr):
        if self.landmarks is None:
            if self.use_oracle_landmarks:
                self.landmarks = self.oracle_landmarks
            else:
                self.landmarks = Landmarks(self.max_landmarks, self.add_threshold, self.landmark_paths)
        elif self.landmarks.num_landmarks:
            if (itr + 1) % self.landmark_update_interval == 0:
                observation = self.landmarks.observations

                model_inputs = buffer_to(observation,
                    device=self.device)
                features = self.idf_model(model_inputs, mode='encode')
                self.landmarks.set_features(features)

                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')
                self.landmarks.set_dsr(dsr)

                # self.landmarks.prune_landmarks()
                self.explore = True

    def landmark_mode(self):
        if self.landmarks is not None and self.landmarks.num_landmarks > 0 and self.explore and \
                self._mode != 'eval' and (self.landmark_mode_steps % self.landmark_mode_interval) == 0:

                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None

                inverse_visitations = 1. / np.clip(self.landmarks.visitations, 1, None)
                landmark_probabilities = inverse_visitations / inverse_visitations.sum()
                self.goal_landmark = np.random.choice(range(len(landmark_probabilities)), p=landmark_probabilities)

    def generate_graph(self):
        if self.true_distance:
            self.landmarks.generate_true_graph(self.env_true_dist, self.edge_threshold)
        else:
            self.landmarks.generate_graph()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        if self.landmarks is not None:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            # only add landmarks in explore phase
            if self.explore and self._mode != 'eval' and not self.use_oracle_landmarks:
                self.landmarks.add_landmark(observation, features, dsr)

            if not self.explore:
                if self.current_landmark is None:
                    norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                    landmark_similarity = torch.matmul(self.landmarks.norm_dsr, norm_dsr.T)
                    self.current_landmark = landmark_similarity.argmax().item()

                    cur_x, cur_y = get_true_pos(observation.squeeze())

                    # check if correct starting landmark
                    closest_landmark = None
                    min_dist = None
                    for i, pos in enumerate(self.landmarks.get_pos()):
                        dist = self.env_true_dist[cur_x, cur_y, pos[0], pos[1]]
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            closest_landmark = i

                        if i == self.current_landmark:
                            chosen_landmark_dist = dist

                    self.num_paths += 1
                    if self.current_landmark == closest_landmark:
                        self.correct_start_landmark += 1

                    if chosen_landmark_dist != 0:
                        self.dist_ratio_start_landmark.append(min_dist / chosen_landmark_dist)
                    else:
                        self.dist_ratio_start_landmark.append(1)

                    # TODO: Hack to give the correct starting landmark
                    self.current_landmark = closest_landmark

                    self.generate_graph()
                    self.path = self.landmarks.generate_path(self.current_landmark, self.goal_landmark)
                    self.path_idx = 0

                    if self._mode != 'eval':
                        self.path_freq[:len(self.path)] += 1

                    landmark_x, landmark_y = self.landmarks.get_pos()[self.goal_landmark]
                    self.total_start_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                    if self.total_start_distance == 0:
                        self.total_start_distance = 1
                    
                    self.start_distance = self.total_start_distance

                # Loop until we find a landmark we are not "nearby"
                find_next_landmark = True
                while find_next_landmark:
                    if self.landmark_steps < self.steps_per_landmark:

                        norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                        subgoal_similarity = torch.matmul(self.landmarks.norm_dsr[self.current_landmark], norm_dsr.T)
                        if subgoal_similarity > self.reach_threshold and self._mode != 'eval':
                            self.path_progress[self.path_idx] += 1
                        
                        # TODO: Hack to check if landmark reached based on true manhattan distance
                        cur_x, cur_y = get_true_pos(observation.squeeze())
                        landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                        ending_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

                        if self.current_landmark == self.goal_landmark:
                            reach_threshold = 0
                        else:
                            reach_threshold = self.steps_for_true_reach

                        if ending_distance <= reach_threshold:
                            self.landmarks.visitations[self.current_landmark] += 1
                            
                            if self._mode != 'eval':
                                # TODO: Bucket by starting distance instead
                                self.end_start_dist_progress[self.path_idx].append(float(ending_distance / self.start_distance))
                                self.true_path_progress[self.path_idx] += 1

                            if self.current_landmark == self.goal_landmark:
                                self.explore = True
                                if self._mode == 'eval':
                                    self.eval_end_pos[(cur_x, cur_y)].append(self.current_landmark)
                                    self.eval_distances.append(ending_distance)
                                else:
                                    # TODO: Bucket by starting distance
                                    self.end_start_dist_ratio.append(float(ending_distance / self.total_start_distance))
                                find_next_landmark = False
                            else:
                                self.path_idx += 1
                                self.current_landmark = self.path[self.path_idx]
                                landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                                self.start_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                                find_next_landmark = True
                                self.landmark_steps = 0
                        else:
                            find_next_landmark = False
                    
                    else:
                        # TODO: Hack to check if landmark reached based on true manhattan distance
                        cur_x, cur_y = get_true_pos(observation.squeeze())
                        landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                        ending_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                        if self._mode != 'eval':
                            # TODO: Bucket by starting distance instead
                            self.end_start_dist_progress[self.path_idx].append(float(ending_distance / self.start_distance))

                        if self.current_landmark == self.goal_landmark:
                            self.explore = True
                            if self._mode == 'eval':
                                self.eval_end_pos[(cur_x, cur_y)].append(self.current_landmark)
                                self.eval_distances.append(ending_distance)
                            else:
                                # TODO: Bucket by starting distance instead
                                self.end_start_dist_ratio.append(float(ending_distance / self.total_start_distance))
                            find_next_landmark = False
                            
                        else:
                            self.path_idx += 1
                            self.current_landmark = self.path[self.path_idx]
                            landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                            self.start_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                            find_next_landmark = True
                            self.landmark_steps = 0

        if self.explore:
            action = torch.randint_like(prev_action, high=self.distribution.dim)
            if self._mode != 'eval':
                self.landmark_mode_steps += 1
        elif self.oracle and self._mode != 'eval':
            cur_x, cur_y = get_true_pos(observation.squeeze())
            landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
            new_pos = [[cur_x + 1, cur_y], [cur_x, cur_y + 1], [cur_x - 1, cur_y], [cur_x, cur_y - 1]]
            min_dist = None
            act = None
            for a, pos in enumerate(new_pos):
                new_x, new_y = pos
                dist = self.env_true_dist[new_x, new_y, landmark_x, landmark_y]
                if dist != -1:
                    if min_dist is None or dist < min_dist:
                        act = a
                        min_dist = dist

            action = torch.zeros_like(prev_action) + act
            self.landmark_steps += 1
        else:
            if self.use_sf:
                subgoal_landmark_dsr = self.landmarks.norm_dsr[self.current_landmark]
                q_values = torch.matmul(dsr, subgoal_landmark_dsr).cpu()
            else:
                subgoal_landmark_features = self.landmarks.norm_features[self.current_landmark]
                q_values = torch.matmul(dsr, subgoal_landmark_features).cpu()
            if self.use_soft_q:
                prob = F.softmax(q_values, dim=1)
                action = self.soft_distribution.sample(DistInfo(prob=prob))
            else:
                action = self.distribution.sample(q_values)
            self.landmark_steps += 1

        self.landmark_mode()

        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def set_eval_goal(self, goal_obs):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            if self.use_oracle_landmarks:
                pass
            elif self.use_oracle_eval_landmarks:
                self.landmarks_storage = self.landmarks
                self.landmarks = self.oracle_landmarks
            else:
                observation = torchify_buffer(goal_obs).unsqueeze(0).float()

                model_inputs = buffer_to(observation,
                        device=self.device)
                features = self.idf_model(model_inputs, mode='encode')

                model_inputs = buffer_to(features,
                        device=self.device)
                dsr = self.model(model_inputs, mode='dsr')

                self.landmarks.force_add_landmark(observation, features, dsr)

            self.first_reset = True
            self.eval_distances = []
            self.eval_end_pos = defaultdict(list)
            return True
        else:
            return False

    def reset(self):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            if self._mode == 'eval':
                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None
                self.goal_landmark = self.landmarks.num_landmarks - 1
            else:
                self.landmark_mode_steps = 0
                self.explore = True

    def reset_one(self, idx):
        self.reset()

    @torch.no_grad()
    def remove_eval_goal(self):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            if self.use_oracle_landmarks:
                pass
            elif self.use_oracle_eval_landmarks:
                self.landmarks = self.landmarks_storage
            else:
                self.landmarks.force_remove_landmark()

            self.eval_goal = False
            self.explore = True
