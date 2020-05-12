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
    # Get true (x, y) position of agent from observation
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
        # Add landmark while ignoring similarity thresholds and max landmarks
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
        # Remove last landmark
        save_idx = range(self.num_landmarks - 1)

        self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]
        
        self.visitations = self.visitations[save_idx]

        self.num_landmarks -= 1

    def add_landmark(self, observation, features, dsr):
        # First landmark
        if self.num_landmarks == 0:
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.visitations = np.array([0])
            self.num_landmarks += 1

            self.landmark_adds += 1
        else:
            norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

            # Candidate under similarity threshold w.r.t. existing landmarks
            if sum(similarity < self.threshold) >= (self.num_landmarks - 1):
                self.landmark_adds += 1

                # Add landmark
                if self.num_landmarks < self.max_landmarks:
                    self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks)
                    self.num_landmarks += 1
                    self.visitations = np.append(self.visitations, 0)

                # Replace existing landmark
                else:
                    # Find two landmarks most similar to each other, select one most similar to candidate
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
        # Set/add features of new landmark at idx
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
        # Set/add DSR of new landmark at idx 
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
        # Get true positions of landmarks
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

        # Remove edges with distance > edge threshold
        try_distances = landmark_distances.copy()
        if edge_threshold is not None:
            non_edges = try_distances > edge_threshold
            try_distances[try_distances > edge_threshold] = 0

        # Augment G with edges until it is connected
        G = nx.from_numpy_array(try_distances)
        if not nx.is_connected(G):
            avail = {index: x for index, x in np.ndenumerate(landmark_distances) if non_edges[index]}
            for edge in nx.k_edge_augmentation(G, 1, avail):
                try_distances[edge] = landmark_distances[edge]
            G = nx.from_numpy_array(try_distances)

        self.graph = G
        return self.graph
        
    def generate_graph(self):
        # Generate landmark graph using (1 - similarity) between DSR of landmarks as edge weights 
        n_landmarks = len(self.norm_dsr)
        similarities = torch.clamp(torch.matmul(self.norm_dsr, self.norm_dsr.T), min=-1.0, max=1.0)
        similarities = similarities.detach().cpu().numpy()
        similarities[range(n_landmarks), range(n_landmarks)] = -2
        max_idx = similarities.argmax(axis=1)

        # Remove edges with similarity < edge threshold
        landmark_distances = 1.001 - similarities
        non_edges = similarities < self.threshold
        non_edges[range(n_landmarks), max_idx] = False
        landmark_distances[non_edges] = 0

        # Augment G with edges until it is connected
        G = nx.from_numpy_array(landmark_distances)
        if not nx.is_connected(G):
            avail = {index: 1.001 - x for index, x in np.ndenumerate(similarities) if non_edges[index]}
            for edge in nx.k_edge_augmentation(G, 1, avail):
                landmark_distances[edge] = 1.001 - similarities[edge]
            G = nx.from_numpy_array(landmark_distances)

        self.graph = G
        return self.graph

    def generate_path(self, source, target):
        # Generate path from source to target in landmark graph
        
        # Select path with probability given by softmin of path lengths
        if self.landmark_paths is not None:
            paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, source, target, weight='weight'), self.landmark_paths))
            path_lengths = np.array([len(path) for path in paths])
            path_choice = np.random.choice(list(range(len(paths))), p=softmax(-1 * path_lengths))
            self.path = paths[path_choice]
        
        # Select shortest path that has most number of actual nodes
        else:
            max_length = 0
            for path in nx.all_shortest_paths(self.graph, source, target, weight='weight'):
                if len(path) > max_length:
                    self.path = path
                    max_length = len(path)
        return self.path

    def prune_landmarks(self):
        # Prune landmarks that do not meet similarity requirement [NOT CURRENTLY USED]
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
            landmark_update_interval=int(5e3),
            add_threshold=0.75,
            max_landmarks=8,
            edge_threshold=25,
            landmark_paths=None, 
            landmark_mode_interval=100,
            steps_per_landmark=25,
            reach_threshold=0.95,
            use_sf=False,
            use_soft_q=False,
            true_distance=False,
            steps_for_true_reach=2,
            oracle=False,
            use_oracle_landmarks=False,
            use_oracle_eval_landmarks=False,
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
        self.soft_distribution = Categorical(dim=env_spaces.action.n)

    def reset_logging(self):
        # Percentage of times we select the correct start landmark
        self.correct_start_landmark = 0
        self.total_landmark_paths = 0

        # Distance to algo-chosen start landmark / correct start landmark
        self.dist_ratio_start_landmark = []

        # Percentage of times we reach the ith landmark
        self.landmark_attempts = np.zeros(self.max_landmarks + 1)
        self.landmark_reaches = np.zeros(self.max_landmarks + 1)
        self.landmark_true_reaches = np.zeros(self.max_landmarks + 1)

        # End / start distance to ith landmark
        self.landmark_dist_completed = [[] for _ in range(self.max_landmarks + 1)]

        # End / start distance to goal landmark
        self.goal_landmark_dist_completed = []

        if self.landmarks is not None:
            self.landmarks.reset_logging()

    def set_oracle_landmarks(self, env):
        # Set oracle landmarks hardcoded in environment
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
        # Set map of true distances between any two points in environment
        self.env_true_dist = env.get_true_distances()

    def update_landmarks(self, itr):
        # Create Landmarks object if it does not exist
        if self.landmarks is None:
            if self.use_oracle_landmarks:
                self.landmarks = self.oracle_landmarks
            else:
                self.landmarks = Landmarks(self.max_landmarks, self.add_threshold, self.landmark_paths)
        
        # Update features and DSR of existing landmarks
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
        # Enter landmark mode during training every landmark_mode_interval steps
        if self.landmarks is not None and self.landmarks.num_landmarks > 0 and self.explore and \
            self._mode != 'eval' and (self.landmark_mode_steps % self.landmark_mode_interval) == 0:

                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None

                # Select goal landmark with probability given by inverse of visitation count
                inverse_visitations = 1. / np.clip(self.landmarks.visitations, 1, None)
                landmark_probabilities = inverse_visitations / inverse_visitations.sum()
                self.goal_landmark = np.random.choice(range(len(landmark_probabilities)), p=landmark_probabilities)

    def generate_graph(self):
        # Use true distance edges
        if self.true_distance:
            self.landmarks.generate_true_graph(self.env_true_dist, self.edge_threshold)

        # Use DSR similarity edges
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

            # Add landmarks if in exploration mode during training
            if self.explore and self._mode != 'eval' and not self.use_oracle_landmarks:
                self.landmarks.add_landmark(observation, features, dsr)

            # Select current (subgoal) landmark
            if not self.explore:

                # Select start landmark based on DSR similarity to current state
                if self.current_landmark is None:
                    norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                    landmark_similarity = torch.matmul(self.landmarks.norm_dsr, norm_dsr.T)
                    self.current_landmark = landmark_similarity.argmax().item()

                    cur_x, cur_y = get_true_pos(observation.squeeze())

                    # Find correct start landmark based on true distances
                    closest_landmark = None
                    min_dist = None
                    for i, pos in enumerate(self.landmarks.get_pos()):
                        dist = self.env_true_dist[cur_x, cur_y, pos[0], pos[1]]
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            closest_landmark = i

                        if i == self.current_landmark:
                            chosen_landmark_dist = dist

                    # Log if selected landmark is correct
                    self.total_landmark_paths += 1
                    if self.current_landmark == closest_landmark:
                        self.correct_start_landmark += 1

                    # Log ratio of distance to selected landmark / correct landmark
                    if chosen_landmark_dist != 0:
                        self.dist_ratio_start_landmark.append(min_dist / chosen_landmark_dist)
                    else:
                        self.dist_ratio_start_landmark.append(1)

                    # HACK: Use correct start landmark
                    self.current_landmark = closest_landmark

                    # Generate landmark graph and path between start and goal landmarks
                    self.generate_graph()
                    self.path = self.landmarks.generate_path(self.current_landmark, self.goal_landmark)
                    self.path_idx = 0

                    # Log that we attempted to reach each landmark in the path
                    if self._mode != 'eval':
                        self.landmark_attempts[:len(self.path)] += 1

                    # Starting distance to goal landmark
                    landmark_x, landmark_y = self.landmarks.get_pos()[self.goal_landmark]
                    self.start_distance_to_goal = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                    if self.start_distance_to_goal == 0:
                        self.start_distance_to_goal = 1
                    
                    self.start_distance_to_landmark = self.start_distance_to_goal

                # Loop on landmarks in path until we find one we have not reached in our current state
                find_next_landmark = True
                while find_next_landmark:

                    # If we still have steps remaining to try to reach the current landmark
                    if self.landmark_steps < self.steps_per_landmark:

                        # Log if current landmark is reached based on similarity
                        norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                        landmark_similarity = torch.matmul(self.landmarks.norm_dsr[self.current_landmark], norm_dsr.T)
                        if landmark_similarity > self.reach_threshold and self._mode != 'eval':
                            self.landmark_reaches[self.path_idx] += 1
                        
                        # HACK: Check if current landmark is reached based on true distance
                        cur_x, cur_y = get_true_pos(observation.squeeze())
                        landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                        end_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

                        # If goal landmark, we must try to reach it perfectly
                        if self.current_landmark == self.goal_landmark:
                            reach_threshold = 0
                        
                        # Otherwise, try to reach the current landmark within a true distance threshold
                        else:
                            reach_threshold = self.steps_for_true_reach

                        # If we have reached the current landmark based on our criteria
                        if end_distance <= reach_threshold:

                            # Increment the current landmark's visitation count
                            self.landmarks.visitations[self.current_landmark] += 1

                            # In training, log that landmark is truly reached andend/start distance to landmark ratio 
                            if self._mode != 'eval':
                                # TODO: Bucket by starting distance instead
                                self.landmark_true_reaches[self.path_idx] += 1
                                self.landmark_dist_completed[self.path_idx].append(float(end_distance / self.start_distance))

                            # If current landmark is goal, exit landmark mode
                            if self.current_landmark == self.goal_landmark:
                                self.explore = True
                                
                                # In eval, log end position trying to reach goal and distance away from goal
                                if self._mode == 'eval':
                                    self.eval_end_pos[(cur_x, cur_y)].append(self.current_landmark)
                                    self.eval_distances.append(end_distance)
                                
                                # In train, log end/start distance to goal ratio
                                else:
                                    # TODO: Bucket by starting distance
                                    self.goal_landmark_dist_completed.append(end_distance / self.start_distance_to_goal)
                                find_next_landmark = False
                            
                            # Else, move to next landmark and set as new subgoal
                            else:
                                self.path_idx += 1
                                self.current_landmark = self.path[self.path_idx]
                                self.landmark_steps = 0
                                find_next_landmark = True

                                landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                                self.start_distance_to_landmark = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]
                        
                        # Otherwise, we continue to use subgoal policy to try to reach current landmark
                        else:
                            find_next_landmark = False
                    
                    # We have run out of steps trying to reach current landmark
                    else:
                        # HACK: Get true distance to current landmark
                        cur_x, cur_y = get_true_pos(observation.squeeze())
                        landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                        end_distance = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

                        # In training, log end/start distance to landmark ratio
                        if self._mode != 'eval':
                            # TODO: Bucket by starting distance instead
                            self.landmark_dist_completed[self.path_idx].append(end_distance / self.start_distance)

                        # If current landmark is goal, exit landmark mode
                        if self.current_landmark == self.goal_landmark:
                            self.explore = True

                            # In eval, log end position trying to reach goal and distance away from goal
                            if self._mode == 'eval':
                                self.eval_end_pos[(cur_x, cur_y)].append(self.current_landmark)
                                self.eval_distances.append(end_distance)

                            # In train, log end/start distance to goal ratio
                            else:
                                # TODO: Bucket by starting distance instead
                                self.goal_landmark_dist_completed.append(end_distance / self.start_distance_to_goal)
                            find_next_landmark = False
                        
                        # Else, move on to next landmark and set as new goal
                        else:
                            self.path_idx += 1
                            self.current_landmark = self.path[self.path_idx]
                            self.landmark_steps = 0
                            find_next_landmark = True

                            landmark_x, landmark_y = self.landmarks.get_pos()[self.current_landmark]
                            self.start_distance_to_landmark = self.env_true_dist[cur_x, cur_y, landmark_x, landmark_y]

        # Exploration (random) policy
        if self.explore:
            action = torch.randint_like(prev_action, high=self.distribution.dim)

            # In training, increment step counter used to track when to enter landmark mode
            if self._mode != 'eval':
                self.landmark_mode_steps += 1

        # Oracle landmark mode (in training only)
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
        
        # Q landmark mode
        else:
            norm_dsr = dsr / torch.norm(dsr, p=2, dim=2, keepdim=True)

            # Use SF-based subgoal Q 
            if self.use_sf:
                subgoal_landmark_dsr = self.landmarks.norm_dsr[self.current_landmark]
                q_values = torch.matmul(norm_dsr, subgoal_landmark_dsr).cpu()
            
            # Use feature-based subgoal Q
            else:
                subgoal_landmark_features = self.landmarks.norm_features[self.current_landmark]
                q_values = torch.matmul(norm_dsr, subgoal_landmark_features).cpu()
            
            # Select action based on softmax of Q as probabilities
            if self.use_soft_q:
                prob = F.softmax(q_values, dim=1)
                action = self.soft_distribution.sample(DistInfo(prob=prob))
            
            # Select action based on epsilon-greedy of Q
            else:
                action = self.distribution.sample(q_values)
            self.landmark_steps += 1

        # Try to enter landmark mode
        self.landmark_mode()

        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def set_eval_goal(self, goal_obs):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            # Using oracle landmarks, which includes goal as landmark
            if self.use_oracle_landmarks:
                pass
            
            # Using oracle landmarks in eval only, need to store training landmarks
            elif self.use_oracle_eval_landmarks:
                self.landmarks_storage = self.landmarks
                self.landmarks = self.oracle_landmarks
            
            # Add goal as landmark for eval only
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
            self.eval_end_pos = defaultdict(list)
            self.eval_distances = []
            return True
        else:
            return False

    def reset(self):
        if self.landmarks and self.landmarks.num_landmarks > 0:
            # In eval, always use landmarks mode
            if self._mode == 'eval':
                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None
                self.goal_landmark = self.landmarks.num_landmarks - 1
            
            # In training, start in exploration mode
            else:
                self.landmark_mode_steps = 0
                self.explore = True

    def reset_one(self, idx):
        self.reset()

    @torch.no_grad()
    def remove_eval_goal(self):
        # Clean up evaluation mode
        if self.landmarks and self.landmarks.num_landmarks > 0:
            if self.use_oracle_landmarks:
                pass
            elif self.use_oracle_eval_landmarks:
                self.landmarks = self.landmarks_storage
            else:
                self.landmarks.force_remove_landmark()

            self.eval_goal = False

            # TODO: Re-continue landmark mode in training that was interrupted by evaluation
            self.explore = True
