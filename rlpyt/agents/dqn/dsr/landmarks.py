from collections import defaultdict
import itertools
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.special import softmax
import torch

from rlpyt.utils.quick_args import save__init__args

SIM_THRESHOLD_CHANGE = 0.1

def euclidean_distance(first_point, second_point):
    return np.sqrt((first_point[0] - second_point[0]) ** 2 +(first_point[1] - second_point[1]) ** 2)

class Landmarks(object):

    def __init__(self,
                 max_landmarks=20,
                 add_threshold=0.75,
                 top_k_similar=None,
                 landmarks_per_update=None,
                 landmark_mode_interval=100,
                 steps_per_landmark=10,
                 max_landmark_mode_steps=500,
                 success_threshold=0,
                 max_attempt_threshold=10,
                 attempt_percentile_threshold=10,
                 sim_threshold=None,
                 sim_percentile_threshold=None,
                 use_oracle_start=False,
                 landmark_paths=1,
                 reach_threshold=0.99,
                 affinity_decay=0.9,
                 oracle_edge_threshold=7
                ):
        save__init__args(locals())
        self.num_landmarks = 0
        self.observations = None 
        self.features = None
        self.norm_features = None
        self.dsr = None
        self.norm_dsr = None
        self.visitations = None
        self.positions = None

        self.potential_landmarks = {}
        if top_k_similar is None:
            self.top_k_similar = max(int(max_landmarks * 0.10), 1)

        if landmarks_per_update is None:
            self.landmarks_per_update = max(int(max_landmarks * 0.10), 1)

        self.predecessors = None
        self.graph = None

        self.successes = None
        self.attempts = None
        self.interval_successes = None
        self.interval_attempts = None
        self.zero_edge_indices = None

        self.potential_landmark_adds = 0
        self._active = False

        self.current_edge_threshold = self.sim_threshold if self.sim_threshold is not None else sim_percentile_threshold
        self.current_sim_threshold = 0
    
    def initialize(self, num_envs, mode='train'):
        self.num_envs = num_envs
        self.mode = mode

        # Landmark mode metadata
        self.landmark_mode = np.full(self.num_envs, mode == 'eval', dtype=bool)
        self.explore_steps = np.full(self.num_envs, 0, dtype=int)
        self.landmark_steps = np.full(self.num_envs, 0, dtype=int)
        self.current_landmark_steps = np.full(self.num_envs, 0, dtype=int)
        self.paths = np.full((self.num_envs, self.max_landmarks), -1, dtype=int)
        self.path_lengths = np.full(self.num_envs, -1, dtype=int)
        self.path_idxs = np.full(self.num_envs, -1, dtype=int)
        self.last_landmarks = np.full(self.num_envs, -1, dtype=int)
        self.start_positions = np.full((self.num_envs, 2), -1, dtype=int)
        self.entered_landmark_mode = np.full(self.num_envs, False, dtype=bool)

        self.eval_end_pos = {}
        self.eval_distances = []

        self.reset_logging()

    def __bool__(self):
        return self._active

    def activate(self):
        self._active = True

    def reset_logging(self):
        # Reset trackers for logging (every interval)

        # Number of landmarks added/removed
        self.landmark_adds = 0
        self.landmark_removes = 0

        # Percentage of times we select the correct start landmark
        self.correct_start_landmark = 0
        self.total_landmark_paths = 0

        # Distance to algo-chosen start landmark metrics
        self.dist_start_landmark = []
        self.dist_ratio_start_landmark = []

        # Attempts used to generate fully connected landmark graph
        self.generate_graph_attempts = [] 

        # Success rates of oracle edges
        self.success_rates = None
        self.interval_success_rates = None
        if self.interval_successes is not None and self.interval_attempts is not None:
            self.interval_successes.fill(0)
            self.interval_attempts.fill(0)

        # Percentage of times we reach the ith landmark
        self.landmark_attempts = np.zeros(self.max_landmarks)
        # self.landmark_reaches = np.zeros(self.max_landmarks)
        self.landmark_true_reaches = np.zeros(self.max_landmarks)

        # End / start distance to goal landmark
        self.goal_landmark_dist_completed = []

    def save(self, filename):
        # Save landmarks data to a file
        np.savez(filename,
                # observations=self.observations.cpu().detach().numpy(),
                features=self.features.cpu().detach().numpy(),
                dsr=self.dsr.cpu().detach().numpy(),
                visitations=self.visitations,
                positions=self.positions,
                successes=self.successes,
                attempts=self.attempts)

    def update(self):
        # Decay empirical transition data by affinity_decay factor
        self.successes = (self.successes * self.affinity_decay)
        self.attempts = (self.attempts * self.affinity_decay)

        # self.interval_successes = (self.interval_successes * self.affinity_decay)
        # self.interval_attempts = (self.interval_attempts * self.affinity_decay)

        # # Prune landmarks that do not meet similarity requirement
        # landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
        # save_idx = torch.sum(landmark_similarities < self.add_threshold, axis=1) >= (self.num_landmarks - 1)
        
        # self.observations = self.observations[save_idx]
        # self.features = self.features[save_idx]
        # self.norm_features = self.norm_features[save_idx]
        # self.dsr = self.dsr[save_idx]
        # self.norm_dsr = self.norm_dsr[save_idx]
        # self.positions = self.positions[save_idx]
        
        # self.visitations = self.visitations[save_idx]
        # self.successes = self.successes[save_idx[:, None], save_idx]
        # self.attempts = self.attempts[save_idx[:, None], save_idx]

        # self.landmark_removes += (self.num_landmarks - len(save_idx))
        # self.num_landmarks = len(save_idx)

    def force_add_landmark(self, features, dsr, position):
        # Add landmark while ignoring similarity thresholds and max landmark limits
        if self.num_landmarks == 0:
            # self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.positions = np.expand_dims(position, 0)
            self.visitations = np.array([0])
            self.successes = np.array([[0]])
            self.attempts = np.array([[0]])
            self.interval_successes = np.array([[0]])
            self.interval_attempts = np.array([[0]])
            self.num_landmarks += 1
        else:
            # self.observations = torch.cat((self.observations, observation), dim=0)
            self.set_features(features, self.num_landmarks)
            self.set_dsr(dsr, self.num_landmarks)
            self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            
            self.visitations = np.append(self.visitations, 0)

            self.successes = np.append(self.successes, np.zeros((self.num_landmarks, 1)), axis=1)
            self.successes = np.append(self.successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
            self.attempts = np.append(self.attempts, np.zeros((self.num_landmarks, 1)), axis=1)
            self.attempts = np.append(self.attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)
            self.interval_successes = np.append(self.interval_successes, np.zeros((self.num_landmarks, 1)), axis=1)
            self.interval_successes = np.append(self.interval_successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
            self.interval_attempts = np.append(self.interval_attempts, np.zeros((self.num_landmarks, 1)), axis=1)
            self.interval_attempts = np.append(self.interval_attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)

            self.num_landmarks += 1

    def force_remove_landmark(self):
        # Remove last landmark
        save_idx = range(self.num_landmarks - 1)

        # self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]

        self.positions = self.positions[save_idx]
        self.visitations = self.visitations[save_idx]

        self.num_landmarks -= 1
        self.successes = self.successes[:self.num_landmarks, :self.num_landmarks]
        self.attempts = self.attempts[:self.num_landmarks, :self.num_landmarks]
        self.interval_successes = self.interval_successes[:self.num_landmarks, :self.num_landmarks]
        self.interval_attempts = self.interval_attempts[:self.num_landmarks, :self.num_landmarks]

    def add_potential_landmark(self, features, dsr, position):
        if self.num_landmarks > 0:
            norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

            # Potential landmarks under similarity threshold w.r.t. existing landmarks
            potential_idxs = torch.sum(similarity < self.add_threshold, dim=0) >= self.num_landmarks
            potential_idxs = potential_idxs.cpu().numpy()
        else:
            potential_idxs = np.ones(len(features), dtype=bool)

        if np.any(potential_idxs):
            if self.potential_landmarks:
                self.potential_landmarks['features'] = torch.cat((self.potential_landmarks['features'],
                                                                  features[potential_idxs]), dim=0)
                self.potential_landmarks['positions'] = np.append(self.potential_landmarks['positions'],
                                                                  position[potential_idxs], axis=0)
            else:
                self.potential_landmarks['features'] = features[potential_idxs].clone()
                self.potential_landmarks['positions'] = position[potential_idxs].copy()
        
        self.potential_landmark_adds += sum(potential_idxs)

    def add_landmarks(self, features, dsr, position):
        # Add landmarks if it is not similar w.r.t existing landmarks
        # by selecting from pool of potential landmarks
        # norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
        # similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        # k_nearest_similarity = torch.topk(similarity, min(self.top_k_similar, self.num_landmarks), dim=0, largest=False).values
        # similarity_score = torch.sum(k_nearest_similarity, dim=0)
        # new_landmark_idxs = torch.topk(similarity_score, min(self.landmarks_per_update, len(observation)), largest=False).indices

        landmarks_added = 0

        for idx in range(len(features)):
            added = self.add_landmark(features[[idx]], dsr[[idx]], position[idx])
            landmarks_added += int(added)
            if landmarks_added == self.landmarks_per_update:
                break

        # # Dynamically adjust add threshold depending on value of self.potential_landmark_adds
        # if self.potential_landmark_adds > self.max_landmarks:
        #     self.add_threshold -= 0.05
        # elif self.potential_landmark_adds == 0:
        #     self.add_threshold += 0.01
        
        self.potential_landmarks = {}
        self.potential_landmark_adds = 0

    def add_landmark(self, features, dsr, position):
        # Add landmark if it is not similar w.r.t. existing landmarks
        if self.num_landmarks == 0:
            # First landmark
            # self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.positions = np.expand_dims(position, 0)
            self.visitations = np.array([0])
            self.successes = np.array([[0]])
            self.attempts = np.array([[0]])
            self.interval_successes = np.array([[0]])
            self.interval_attempts = np.array([[0]])
            self.landmark_adds += 1
            self.num_landmarks += 1
            return True
        else:
            
            norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) # Current SF (A x 512) --> 512, mean over the actions and norm
            # self.norm_dsr: |num landmarks| x 512
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T) # Compute similarity w.r.t. each existing landmark |num landmarks|

            # Candidate landmark under similarity threshold w.r.t. existing landmarks
            if torch.sum(similarity < self.add_threshold) >= self.num_landmarks:
                self.landmark_adds += 1

                # Add landmark
                if self.num_landmarks < self.max_landmarks:
                    # self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks)
                    self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            
                    self.visitations = np.append(self.visitations, 0)
                    
                    self.successes = np.append(self.successes, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.successes = np.append(self.successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
                    self.attempts = np.append(self.attempts, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.attempts = np.append(self.attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)
                    self.interval_successes = np.append(self.interval_successes, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.interval_successes = np.append(self.interval_successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
                    self.interval_attempts = np.append(self.interval_attempts, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.interval_attempts = np.append(self.interval_attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.landmark_adds += 1
                    self.num_landmarks += 1

                # Replace existing landmark
                else:
                    # # Find two landmarks most similar to each other, select one most similar to candidate
                    # landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)

                    # # Do not replace initial landmarks (first two indices)
                    # landmark_similarities[0:2, :] = -2
                    # landmark_similarities[:, 0:2] = -2
                    # landmark_similarities[range(self.num_landmarks), range(self.num_landmarks)] = -2
                    # idx = landmark_similarities.argmax().item()
                    # a, b = (idx // self.num_landmarks), (idx % self.num_landmarks)
                    # if similarity[a] > similarity[b]:
                    #     replace_idx = a
                    # else:
                    #     replace_idx = b

                    success_rates = self.successes / np.clip(self.attempts, 1, None)
                    landmark_success_rates = success_rates.max(axis=0)
                    landmark_attempts = -1 * self.attempts.sum(axis=0)

                    candidates = list(zip(landmark_success_rates[2:], landmark_attempts[2:], range(2, self.num_landmarks)))
                    replace_idx = sorted(candidates)[0][2] 

                    # self.observations[replace_idx] = observation
                    self.set_features(features, replace_idx)
                    self.set_dsr(dsr, replace_idx)
                    self.positions[replace_idx] = np.expand_dims(position, 0)
                    self.visitations[replace_idx] = 0

                    self.successes[replace_idx, :] = 0
                    self.successes[:, replace_idx] = 0
                    self.attempts[replace_idx, :] = 0
                    self.attempts[:, replace_idx] = 0
                    self.interval_successes[replace_idx, :] = 0
                    self.interval_successes[:, replace_idx] = 0
                    self.interval_attempts[replace_idx, :] = 0
                    self.interval_attempts[:, replace_idx] = 0

                    self.landmark_removes += 1
                
                return True

            else:
                return False
            
    def set_features(self, features, idx=None):
        # Set/add features of new landmark at idx
        norm_features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        if self.features is None or idx is None:
            self.features = features
            self.norm_features = norm_features
        elif isinstance(idx, np.ndarray) or (0 <= idx and idx < self.num_landmarks):
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
        elif isinstance(idx, np.ndarray) or (0 <= idx and idx < self.num_landmarks):
            self.dsr[idx] = dsr
            self.norm_dsr[idx] = norm_dsr
        else:
            self.dsr = torch.cat((self.dsr, dsr), dim=0)
            self.norm_dsr = torch.cat((self.norm_dsr, norm_dsr), dim=0)

    def get_oracle_edges(self, env):
        num_rooms = len(env.rooms)
        door_pos = env.get_doors()
        num_doors = len(door_pos)

        def get_door_states(obs):
            door_states = []
            for pos in door_pos:
                if obs[pos[0], pos[1], 0] != 4:
                    door_states.append(-1)
                else:
                    door_states.append(obs[pos[0], pos[1], 2])
            return np.array(door_states, dtype=int)
        
        # get oracle edges
        N = self.num_landmarks
        oracle_edges = np.zeros((N, N), dtype=bool)
        observations = self.observations.cpu().detach().numpy()

        for i in range(N):
            for j in range(i + 1, N):
                pos_i = self.positions[i]
                pos_j = self.positions[j]

                room_i, is_room_i = env.get_room(pos_i)
                room_j, is_room_j = env.get_room(pos_j)

                door_states_i = get_door_states(observations[i])
                door_states_j = get_door_states(observations[j])
                door_states_compared = door_states_i == door_states_j
                num_same_doors = door_states_compared.sum()

                edge_exists = False

                if is_room_i and is_room_j:
                    if room_i == room_j:
                        if num_same_doors == num_doors:
                            edge_exists = True
                        elif num_same_doors == (num_doors - 1):
                            if room_i == 0:
                                edge_exists = not door_states_compared[room_i]
                            elif room_i == (num_rooms - 1):
                                edge_exists = not door_states_compared[room_i - 1]
                            else:
                                edge_exists = door_states_compared[room_i] ^ door_states_compared[room_i - 1]
                    elif abs(room_i - room_j) == 1:
                        distance = self.oracle_distance_matrix[pos_i[0], pos_i[1], pos_j[0], pos_j[1]]
                        if distance < self.oracle_edge_threshold and num_same_doors == num_doors:
                            edge_exists = door_states_i[min(room_i, room_j)] == 0
                elif is_room_i ^ is_room_j:
                    if num_same_doors == (num_doors - 1):
                        if is_room_i:
                            edge_exists = room_i == room_j | (room_i - 1) == room_j
                        else:
                            edge_exists = room_j == room_i | (room_j - 1) == room_i

                oracle_edges[i, j] = edge_exists

        return oracle_edges

    def get_low_attempt_threshold(self, use_max=True):
        if self.num_landmarks > 1:
            threshold = np.percentile(self.attempts[~np.eye(self.num_landmarks, dtype=bool)], self.attempt_percentile_threshold)
        else:
            threshold = self.max_attempt_threshold
        if use_max:
            threshold = min(threshold, self.max_attempt_threshold)
        return threshold
    
    def get_sim_threshold(self, attempt=0, similarity_matrix=None):
        self.current_edge_threshold -= (SIM_THRESHOLD_CHANGE * attempt)
        if self.sim_threshold:
            return self.current_edge_threshold      
        elif self.sim_percentile_threshold and similarity_matrix is not None:
            return np.percentile(similarity_matrix[~np.eye(self.num_landmarks, dtype=bool)],
                self.current_edge_threshold)
        else:
            raise RuntimeError('Set either sim_threshold or sim_percentile_threshold')

    def generate_graph(self):
        # Generate landmark graph using empirical transitions
        # and similarity in SF space between landmarks
        edge_success_rate = self.successes / np.clip(self.attempts, 1, None)
        self.landmark_distances = edge_success_rate.copy()

        # Distance for edges with no successful transitions is based on minimum success rate
        non_zero_success = edge_success_rate[edge_success_rate > 0]
        if non_zero_success.size == 0:
            zero_success_dist = 1e-3
        else:
            zero_success_dist = non_zero_success.min()

        similarities = torch.clamp(torch.matmul(self.norm_dsr, self.norm_dsr.T), min=1e-3, max=1.0)
        similarities = similarities.detach().cpu().numpy()
        min_dist = zero_success_dist * similarities

        # Remove edges with success rate <= success_threshold
        non_edges = np.logical_not(edge_success_rate > self.success_threshold)

        # # If less than 5% of possible edges have non-zero success rates,
        # # then use edges with high similarity 
        # N = self.num_landmarks
        # if torch.sum(edge_success_rate > 0) < 0.05 * (N * (N - 1) / 2):
        #     high_similarity_edges = similarities > self.sim_threshold
        #     non_edges[high_similarity_edges] = False 
        #     landmark_distances[high_similarity_edges] = np.clip(landmark_distances[high_similarity_edges], a_min=min_dist[high_similarity_edges], a_max=None)

        # In all modes except eval, consider edges with low numbers
        # of attempted transitions as valid starting edges
        if self.mode != 'eval':
            attempt_threshold = self.get_low_attempt_threshold()
            low_attempt_edges = self.attempts <= attempt_threshold
            non_edges[low_attempt_edges] = False
            # low_attempt_dist = 1
            if non_zero_success.size == 0:
                low_attempt_dist = 1e-3 * similarities
            else:
                low_attempt_dist = non_zero_success.mean() * similarities
            self.landmark_distances[low_attempt_edges] = np.clip(self.landmark_distances[low_attempt_edges],
                                                                 a_min=low_attempt_dist[low_attempt_edges], a_max=None)

        # Distance = -1 * np.log (transition probability)
        self.landmark_distances[non_edges] = 0
        self.landmark_distances[self.landmark_distances == 1] = 1 - 1e-6
        self.landmark_distances[self.landmark_distances != 0] = -1 * np.log(self.landmark_distances[self.landmark_distances != 0])

        # # Penalize edges with no success, high attempts
        # attempt_threshold = max(non_zero_attempts.mean() + non_zero_attempts.std(), 1)
        # high_attempt_edges = self.attempts > attempt_threshold
        # edge_success_rate[edge_success_rate == 0 & high_attempt_edges] = -1

        # Augment G with edges until it is connected
        # Edges are sorted by success rate, then similarity
        self.zero_edge_indices = set()
        self.graph = nx.from_numpy_array(self.landmark_distances, create_using=nx.DiGraph)

        attempt = 0
        while not nx.is_strongly_connected(self.graph):
            self.current_sim_threshold = self.get_sim_threshold(attempt, similarities)
            add_edges_by_sim = non_edges & (similarities >= self.current_sim_threshold)
            self.landmark_distances[add_edges_by_sim] = min_dist[add_edges_by_sim]
            self.graph = nx.from_numpy_array(self.landmark_distances, create_using=nx.DiGraph)
            attempt += 1
        
        # If no need to adjust threshold, then try to make it more conservative
        if attempt == 1 and self.current_edge_threshold:
           self.current_edge_threshold += SIM_THRESHOLD_CHANGE
        
        self.generate_graph_attempts.append(attempt)

        # if not nx.is_strongly_connected(self.graph):
        #     avail = []
        #     for index, x in np.ndenumerate(similarities):
        #         if non_edges[index]:
        #             avail.append((edge_success_rate[index], x, *index))
        #     avail = sorted(avail, key=itemgetter(0, 1), reverse=True)

        #     for success_rate, similarity, u, v in avail:
        #         if success_rate > 0:
        #             dist = -1 * np.log(success_rate)
        #         else:
        #             dist = -1 * self.max_landmarks * np.log(zero_success_dist * similarity)

        #         landmark_distances[(u, v)] = dist
                
        #         self.landmark_distances = landmark_distances
        #         self.graph = nx.from_numpy_array(landmark_distances, create_using=nx.DiGraph)

        #         if success_rate <= 0:
        #             self.zero_edge_indices.add((u, v))

        #         if nx.is_strongly_connected(self.graph):
        #             break
        
        return self.graph
    
    def connect_goal(self):
        if self.num_landmarks > 1:
            self.landmark_distances = np.append(self.landmark_distances, np.zeros((self.num_landmarks - 1, 1)), axis=1)
            self.landmark_distances = np.append(self.landmark_distances, np.zeros((1, self.num_landmarks)), axis=0)
            similarity_to_goal = (self.norm_dsr[-1] * self.norm_dsr[:-1]).sum(dim=1)
            closest_to_goal = similarity_to_goal.argmax().item()

            self.landmark_distances[closest_to_goal, -1] = 1
            self.landmark_distances[-1, closest_to_goal] = 1
        else:
            self.landmark_distances = np.ones((1, 1))
        
        goal_index = self.num_landmarks - 1

        self.graph.add_node(goal_index)
        self.graph.add_edge(closest_to_goal, goal_index)
        self.graph.add_edge(goal_index, closest_to_goal)

    def get_path_weight(self, nodes, weight):
        # Get weight of path
        w = 0
        for i, node in enumerate(nodes[1:]):
            prev = nodes[i]
            w += self.graph[prev][node][weight]
        return w

    def generate_path(self, source, target):
        # Generate path from source to target in landmark graph
        
        # Get k shortest paths (k = self.landmark_paths)
        if self.mode == 'eval':
            k = 1
        else:
            k = self.landmark_paths

        if k > 1:
            paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, source, target, weight='weight'), k))
            if self.mode == 'eval':
                # Weights defined by path weight
                path_weights = np.array([self.get_path_weight(path, 'weight') for path in paths])
            else:
                # Weights defined by index of path, where paths sorted by distance
                path_weights = np.arange(1, len(paths) + 1)

            # Select path with probability given by softmin of path weights
            self.possible_paths = paths
            self.path_p = softmax(-1 * path_weights)
            path_choice = np.random.choice(list(range(len(paths))), p=self.path_p)

            return paths[path_choice]
        else:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            self.possible_paths = [path] 
            self.path_p = [1.0]
            return path

    def enter_landmark_mode(self, override=None):
        if override is not None:
            enter_idxs = np.full(self.num_envs, False, dtype=bool)
            if override == -1:
                enter_idxs[:] = True
            else:
                enter_idxs[override] = True
        else:
            enter_idxs = (~self.landmark_mode) & (self.explore_steps > self.landmark_mode_interval) 

        self.landmark_mode[enter_idxs] = True
        self.explore_steps[enter_idxs] = 0
        self.landmark_steps[enter_idxs] = 0 
        self.current_landmark_steps[enter_idxs] = 0
        self.paths[enter_idxs] = -1
        self.path_idxs[enter_idxs] = 0
        self.last_landmarks[enter_idxs] = -1

        self.entered_landmark_mode[enter_idxs] = True
        return self.entered_landmark_mode

    def set_paths(self, dsr, position, relocalize_idxs=None):
        if relocalize_idxs is None:
            set_paths_idxs = self.entered_landmark_mode.copy()
            self.entered_landmark_mode[self.entered_landmark_mode] = False
        else:
            set_paths_idxs = relocalize_idxs

        if not np.any(set_paths_idxs):
            return
        
        norm_dsr = dsr[set_paths_idxs]
        selected_position = position[set_paths_idxs]

        norm_dsr = norm_dsr.mean(dim=1) / torch.norm(norm_dsr.mean(dim=1), p=2, keepdim=True)
        landmark_similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        # Select start landmarks based on SF similarity w.r.t. current observations
        start_landmarks = landmark_similarity.argmax(axis=0).cpu().detach().numpy()

        if relocalize_idxs is None:
            if self.mode == 'eval':
                # Goal landmark is set as the last landmark to be added
                goal_landmarks = np.full(sum(set_paths_idxs), self.num_landmarks - 1, dtype=int)
            else:
                # Select goal landmarks with probability given by inverse of visitation count
                # visitations = np.clip(self.visitations, 1, None)
                visitations = np.clip(self.successes.sum(axis=1), 1, None)

                inverse_visitations = 1. / visitations
                landmark_probabilities = inverse_visitations / inverse_visitations.sum()
                goal_landmarks = np.random.choice(range(len(landmark_probabilities)),
                                                size=len(start_landmarks),
                                                replace=True,
                                                p=landmark_probabilities)
        else:
            prev_start_landmarks = self.paths[relocalize_idxs, 0]
            goal_landmarks = self.paths[relocalize_idxs, self.path_lengths[relocalize_idxs] - 1]
            self.current_landmark_steps[relocalize_idxs] = 0
            self.paths[relocalize_idxs, :] = -1
            self.path_idxs[relocalize_idxs, :] = 0
            self.last_landmarks[relocalize_idxs] = -1

        enter_idxs = np.arange(self.num_envs)[set_paths_idxs]

        for i, enter_idx, start_pos, start_landmark, goal_landmark in zip(range(len(enter_idxs)), enter_idxs, selected_position, start_landmarks, goal_landmarks):
            if relocalize_idxs is not None and start_landmark == prev_start_landmarks[i]:
                continue

            self.start_positions[enter_idx] = start_pos
            cur_x, cur_y = start_pos

            # Find correct start landmark based on true distances
            dist_to_selected_start = euclidean_distance(start_pos, self.positions[start_landmark])
            dist_to_estimated_best_start = np.clip(np.linalg.norm(self.positions - start_pos).min(), 1e-6)

            self.dist_start_landmark.append(dist_to_selected_start)
            self.dist_ratio_start_landmark.append(dist_to_selected_start / dist_to_estimated_best_start)

            if self.oracle_distance_matrix:
                closest_landmark = None
                min_dist = None
                for i, pos in enumerate(self.positions):
                    dist = self.oracle_distance_matrix[cur_x, cur_y, pos[0], pos[1]]
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        closest_landmark = i

                    if i == start_landmark:
                        chosen_landmark_dist = dist

                # Log if selected landmark is correct
                self.total_landmark_paths += 1
                if start_landmark == closest_landmark:
                    self.correct_start_landmark += 1

                # Log ratio of distance to selected landmark / correct landmark
                if chosen_landmark_dist:
                    self.dist_ratio_start_landmark.append(min_dist / chosen_landmark_dist)
                else:
                    self.dist_ratio_start_landmark.append(1)

                # ORACLE: Use correct start landmark
                if self.use_oracle_start:
                    start_landmark = closest_landmark

            path = self.generate_path(start_landmark, goal_landmark)
            path_length = len(path)
            self.paths[enter_idx, :path_length] = path            
            self.path_lengths[enter_idx] = path_length
            
            # # Log that we attempted to reach each landmark in the path
            # if self.mode != 'eval':
            #     self.landmark_attempts[:path_length] += 1

    def log_eval(self, idx, pos):
        # In eval, log end position trying to reach goal and distance away from goal
        current_idx = self.path_idxs[idx]
        current_landmark = self.paths[idx, current_idx]
        self.eval_end_pos[tuple(pos)] = current_landmark

        goal_pos = self.positions[-1]
        if self.oracle_distance_matrix:
            end_distance = self.oracle_distance_matrix[pos[0], pos[1], goal_pos[0], goal_pos[1]]
        else:
            # Use euclidean distance as rough estimate of distane to goal
            end_distance = euclidean_distance(pos[:2], goal_pos[:2])
        self.eval_distances.append(end_distance)

    def get_landmarks_data(self, current_dsr, current_position):
        if not np.any(self.landmark_mode) or self.num_landmarks == 0:
            return None, self.landmark_mode

        current_idxs = self.path_idxs[self.landmark_mode]
        current_landmarks = self.paths[self.landmark_mode, current_idxs]
        final_goal_landmarks = current_idxs == (self.path_lengths[self.landmark_mode] - 1)

        # Localization based on SF similarity
        norm_dsr = current_dsr[self.landmark_mode]
        norm_dsr = norm_dsr.mean(dim=1) / torch.norm(norm_dsr.mean(dim=1), p=2, keepdim=True)
        landmark_similarity = torch.sum(norm_dsr * self.norm_dsr[current_landmarks], dim=1)
        reached_landmarks = landmark_similarity > self.reach_threshold

        # # Localization based on observation equivalence
        # reached_landmarks = self.landmark_mode[self.landmark_mode]
        # for i, observation in enumerate(current_observation[self.landmark_mode]):
        #     reached_landmarks[i] = torch.allclose(observation, self.observations[current_landmarks[i]])

        # for i, observation in enumerate(current_observation[self.landmark_mode]):
        #     if current_landmarks[i] == 0:
        #         reached_landmarks[i] = torch.allclose(observation, self.observations[current_landmarks[i]])

        # If eval mode, keep trying to reach goal until episode terminates
        if self.mode == 'eval':
            reached_landmarks[final_goal_landmarks] = False

        reached_landmarks = reached_landmarks.detach().cpu().numpy() 

        # Increment the current landmark's visitation count
        self.visitations[current_landmarks[reached_landmarks]] += 1

        if self.mode != 'eval':
            steps_limit = np.minimum(self.path_lengths[self.landmark_mode] * self.steps_per_landmark, self.max_landmark_mode_steps)
            steps_limit_reached = self.landmark_steps[self.landmark_mode] >= steps_limit
        else:
            steps_limit_reached = False

        reached_within_steps = self.current_landmark_steps[self.landmark_mode] < self.steps_per_landmark

        # # Relocalize agent which has failed to reach current landmark in steps_per_landmark
        # if self.mode == 'eval':
        #     relocalize_idxs = self.landmark_mode.copy()
        #     relocalize_idxs[self.landmark_mode] &= ~reached_landmarks & ~reached_within_steps & ~final_goal_landmarks
        #     self.set_paths(current_dsr, current_position, relocalize_idxs)
        
        if self.mode != 'eval':
            # # In training, log landmarks are truly reached 
            # reaches = np.bincount(current_landmarks[reached_landmarks])
            # self.landmark_true_reaches[:len(reaches)] += reaches

            # Update landmark transition success rates
            last_landmarks = self.last_landmarks[self.landmark_mode]
            reached_last_landmarks = last_landmarks != -1

            successful_transitions = reached_last_landmarks & reached_landmarks & reached_within_steps
            success_last_landmarks = last_landmarks[successful_transitions]
            success_current_landmarks = current_landmarks[successful_transitions]
            self.successes[success_last_landmarks, success_current_landmarks] += 1
            self.attempts[success_last_landmarks, success_current_landmarks] += 1
            self.interval_successes[success_last_landmarks, success_current_landmarks] += 1
            self.interval_attempts[success_last_landmarks, success_current_landmarks] += 1

            failed_transitions = reached_last_landmarks & ~reached_landmarks & (~reached_within_steps | steps_limit_reached)
            fail_last_landmarks = last_landmarks[failed_transitions]
            fail_current_landmarks = current_landmarks[failed_transitions]
            self.attempts[fail_last_landmarks, fail_current_landmarks] += 1
            self.interval_attempts[fail_last_landmarks, fail_current_landmarks] += 1

        # If reached (not goal) landmark, move to next landmark
        reached_non_goal_landmarks = reached_landmarks & ~final_goal_landmarks
        reached_non_goal_landmarks_mask = np.where(self.landmark_mode)[0][reached_non_goal_landmarks]
        self.last_landmarks[reached_non_goal_landmarks_mask] = current_landmarks[reached_non_goal_landmarks]
        self.current_landmark_steps[reached_non_goal_landmarks_mask] = 0
        self.path_idxs[reached_non_goal_landmarks_mask] += 1

        # If reached goal landmark or steps limit, exit landmark mode
        reached_goal_landmarks = reached_landmarks & final_goal_landmarks 
        end_positions = current_position[self.landmark_mode][reached_goal_landmarks | steps_limit_reached]
        goal_landmarks = self.paths[self.landmark_mode, self.path_lengths[self.landmark_mode] - 1]
        goal_landmarks = goal_landmarks[reached_goal_landmarks | steps_limit_reached]
        goal_positions = self.positions[goal_landmarks]

        if self.mode != 'eval' and self.oracle_distance_matrix:
            # In train, log end/start distance to goal ratio
            end_distance = self.oracle_distance_matrix[end_positions[:, 0], end_positions[:, 1], goal_positions[:, 0], goal_positions[:, 1]]
            start_positions = self.start_positions[self.landmark_mode][reached_goal_landmarks | steps_limit_reached]
            start_distance = self.oracle_distance_matrix[start_positions[:, 0], start_positions[:, 1], goal_positions[:, 0], goal_positions[:, 1]]
            dist_completed = end_distance / np.clip(start_distance, a_min=1, a_max=None)
            self.goal_landmark_dist_completed.extend(dist_completed.tolist())

        exit_landmark_mode = np.where(self.landmark_mode)[0][reached_goal_landmarks | steps_limit_reached]
        self.landmark_mode[exit_landmark_mode] = False

        # Increment landmark step counter
        self.landmark_steps[self.landmark_mode] += 1
        self.current_landmark_steps[self.landmark_mode] += 1

        # In training, increment explore step counter
        if self.mode != 'eval':
            self.explore_steps[~self.landmark_mode] += 1

        current_idxs = self.path_idxs[self.landmark_mode]
        current_landmarks = self.paths[self.landmark_mode, current_idxs]
        return self.norm_dsr[current_landmarks], self.landmark_mode

    def generate_true_graph(self, oracle_distance_matrix, edge_threshold=None):
        # Generate landmark graph using true distances given by oracle
        n_landmarks = len(self.norm_dsr)
        landmark_distances = np.zeros((n_landmarks, n_landmarks))

        for s in range(n_landmarks):
            s_x, s_y = self.positions[s]
            for t in range(n_landmarks):
                t_x, t_y = self.positions[t]
                if s_x == t_x and s_y == t_y:
                    landmark_distances[s, t] = 1
                else:
                    landmark_distances[s, t] = oracle_distance_matrix[s_x, s_y, t_x, t_y]

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
        
