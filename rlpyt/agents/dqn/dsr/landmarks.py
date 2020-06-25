from collections import defaultdict
import itertools
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.special import softmax
import torch

from rlpyt.utils.quick_args import save__init__args


class Landmarks(object):

    def __init__(self,
                 max_landmarks=20,
                 add_threshold=0.75,
                 top_k_similar=None,
                 landmarks_per_update=None,
                 landmark_mode_interval=100,
                 steps_per_landmark=10,
                 success_threshold=0,
                 sim_threshold=0.9,
                 use_oracle_start=False,
                 landmark_paths=1,
                 reach_threshold=0.99,
                 affinity_decay=0.9,
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
        self.zero_edge_indices = None

        self.potential_landmark_adds = 0
        self.reset_logging() 
        self._active = False
    
    def initialize(self, num_envs, mode='train'):
        self.num_envs = num_envs
        self.mode = mode

        # Landmark mode metadata
        self.landmark_mode = np.full(self.num_envs, True, dtype=bool)
        self.explore_steps = np.full(self.num_envs, 0, dtype=int)
        self.landmark_steps = np.full(self.num_envs, 0, dtype=int)
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

        # Distance to algo-chosen start landmark / correct start landmark
        self.dist_ratio_start_landmark = []
        self.success_rates = []

        # Percentage of times we reach the ith landmark
        self.landmark_attempts = np.zeros(self.max_landmarks)
        # self.landmark_reaches = np.zeros(self.max_landmarks)
        self.landmark_true_reaches = np.zeros(self.max_landmarks)

        # End / start distance to goal landmark
        self.goal_landmark_dist_completed = []

        self.non_zero_success_rates = []
        self.zero_success_edge_ratio = []
        
    def save(self, filename):
        # Save landmarks data to a file
        np.savez(filename,
                 observations=self.observations.cpu().detach().numpy(),
                 features=self.features.cpu().detach().numpy(),
                 dsr=self.dsr.cpu().detach().numpy(),
                 visitations=self.visitations,
                 positions=self.positions,
                 successes=self.successes,
                 attempts=self.attempts)

    def force_add_landmark(self, observation, features, dsr, position):
        # Add landmark while ignoring similarity thresholds and max landmark limits
        if self.num_landmarks == 0:
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.positions = np.expand_dims(position, 0)
            self.visitations = np.array([0])
            self.num_landmarks += 1
            self.successes = np.array([[0]])
            self.attempts = np.array([[0]])

        else:
            self.observations = torch.cat((self.observations, observation), dim=0)
            self.set_features(features, self.num_landmarks)
            self.set_dsr(dsr, self.num_landmarks)
            self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            
            self.visitations = np.append(self.visitations, 0)

            self.successes = np.append(self.successes, np.zeros((self.num_landmarks, 1)), axis=1)
            self.successes = np.append(self.successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
            self.attempts = np.append(self.attempts, np.zeros((self.num_landmarks, 1)), axis=1)
            self.attempts = np.append(self.attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)
           
            self.num_landmarks += 1

    def force_remove_landmark(self):
        # Remove last landmark
        save_idx = range(self.num_landmarks - 1)

        self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]

        self.positions = self.positions[save_idx]
        self.visitations = self.visitations[save_idx]

        self.num_landmarks -= 1
        self.successes = self.successes[:self.num_landmarks, :self.num_landmarks]
        self.attempts = self.attempts[:self.num_landmarks, :self.num_landmarks]

    def add_potential_landmark(self, observation, dsr, position):
        potential_observation = observation[self.landmark_mode]
        potential_dsr = dsr[self.landmark_mode]
        potential_position = position[self.landmark_mode]

        norm_dsr = potential_dsr.mean(dim=1) / torch.norm(potential_dsr.mean(dim=1), p=2, keepdim=True)
        similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        # Potential landmarks under similarity threshold w.r.t. existing landmarks
        potential_idxs = torch.sum(similarity < self.add_threshold, dim=0) >= self.num_landmarks
        potential_idxs = potential_idxs.cpu().numpy()

        if np.any(potential_idxs):
            if self.potential_landmarks:
                self.potential_landmarks['observation'] = torch.cat((self.potential_landmarks['observation'],
                                                                     potential_observation[potential_idxs]), dim=0)
                self.potential_landmarks['positions'] = np.append(self.potential_landmarks['positions'],
                                                                  potential_position[potential_idxs], axis=0)
            else:
                self.potential_landmarks['observation'] = potential_observation[potential_idxs].clone()
                self.potential_landmarks['positions'] = potential_position[potential_idxs].copy()
        
        self.potential_landmark_adds += sum(potential_idxs)

    def add_landmarks(self, observation, features, dsr, position):
        # Add landmarks if it is not similar w.r.t existing landmarks
        # by selecting from pool of potential landmarks
        norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
        similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        k_nearest_similarity = torch.topk(similarity, min(self.top_k_similar, self.num_landmarks), dim=0, largest=False).values
        similarity_score = torch.sum(k_nearest_similarity, dim=0)
        new_landmark_idxs = torch.topk(similarity_score, min(self.landmarks_per_update, len(observation)), largest=False).indices

        for idx in new_landmark_idxs:
            idx = idx.item()
            self.add_landmark(observation[[idx]], features[[idx]], dsr[[idx]], position[idx])

        # # Dynamically adjust add threshold depending on value of self.potential_landmark_adds
        # if self.potential_landmark_adds > self.max_landmarks:
        #     self.add_threshold -= 0.05
        # elif self.potential_landmark_adds == 0:
        #     self.add_threshold += 0.01
        
        self.potential_landmarks = {}
        self.potential_landmark_adds = 0

    def update(self):
        # Decay empirical transition data by affinity_decay factor
        self.successes = (self.successes * self.affinity_decay)
        self.attempts = (self.attempts * self.affinity_decay)

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

    def add_landmark(self, observation, features, dsr, position):
        # Add landmark if it is not similar w.r.t. existing landmarks
        if self.num_landmarks == 0:
            # First landmark
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.positions = np.expand_dims(position, 0)
            self.visitations = np.array([0])
            self.successes = np.array([[0]])
            self.attempts = np.array([[0]])
            self.landmark_adds += 1
            self.num_landmarks += 1
        else:
            norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

            # Candidate landmark under similarity threshold w.r.t. existing landmarks
            if sum(similarity < self.add_threshold) >= self.num_landmarks:
                self.landmark_adds += 1

                # Add landmark
                if self.num_landmarks < self.max_landmarks:
                    self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks)
                    self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            
                    
                    self.successes = np.append(self.successes, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.successes = np.append(self.successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
                    self.attempts = np.append(self.attempts, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.attempts = np.append(self.attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.num_landmarks += 1
                    self.visitations = np.append(self.visitations, 0)

                # Replace existing landmark
                else:
                    # Find two landmarks most similar to each other, select one most similar to candidate
                    landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
                    landmark_similarities[0, :] = -2
                    landmark_similarities[:, 0] = -2
                    landmark_similarities[range(self.num_landmarks), range(self.num_landmarks)] = -2
                    idx = landmark_similarities.argmax().item()
                    a, b = (idx // self.num_landmarks), (idx % self.num_landmarks)
                    if similarity[a] > similarity[b]:
                        replace_idx = a
                    else:
                        replace_idx = b
                    self.observations[replace_idx] = observation
                    self.set_features(features, replace_idx)
                    self.set_dsr(dsr, replace_idx)
                    self.positions[replace_idx] = np.expand_dims(position, 0)
                    self.visitations[replace_idx] = 0

                    self.successes[replace_idx, :] = 0
                    self.successes[:, replace_idx] = 0
                    self.attempts[replace_idx, :] = 0
                    self.attempts[:, replace_idx] = 0

                    self.landmark_removes += 1
            
            # Record landmark transitions found during exploration mode
            # TODO: Still in the works!
            # if self.last_landmark is not None and self.last_landmark != current_landmark:
            #     if self.attempts[self.last_landmark, current_landmark]:
            #         new_successes = (self.successes[self.last_landmark, current_landmark] + self.attempts[self.last_landmark, current_landmark]) / 2
            #         self.successes[self.last_landmark, current_landmark] = new_successes
            #         self.successes[current_landmark, self.last_landmark] = new_successes
            #     else:
            #         self.successes[self.last_landmark, current_landmark] = 1
            #         self.successes[current_landmark, self.last_landmark] = 1
            #         self.attempts[self.last_landmark, current_landmark] = 1
            #         self.attempts[current_landmark, self.last_landmark] = 1
            
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

    def generate_graph(self):
        # Generate landmark graph using empirical transitions
        # and similarity in SF space between landmarks
        edge_success_rate = self.successes / np.clip(self.attempts, 1, None)
        landmark_distances = edge_success_rate.copy()

        # Distance for edges with no successful transitions is based on minimum success rate
        non_zero_success = edge_success_rate[edge_success_rate > 0]
        if non_zero_success.size == 0:
            zero_success_dist = 1e-3
        else:
            zero_success_dist = non_zero_success.min()

        similarities = torch.clamp(torch.matmul(self.norm_dsr, self.norm_dsr.T), min=1e-2, max=1.0)
        similarities = similarities.detach().cpu().numpy()
        min_dist = similarities * zero_success_dist

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
            non_zero_attempts = self.attempts[self.attempts > 0]
            attempt_threshold = max(non_zero_attempts.mean() - non_zero_attempts.std(), 1)
            low_attempt_edges = self.attempts < attempt_threshold
            non_edges[low_attempt_edges] = False            
            landmark_distances[low_attempt_edges] = np.clip(landmark_distances[low_attempt_edges], a_min=min_dist[low_attempt_edges], a_max=None)

        # Distance = -1 * np.log (transition probability)
        landmark_distances[non_edges] = 0
        landmark_distances[landmark_distances == 1] = 1 - 1e-6
        landmark_distances[landmark_distances != 0] = -1 * np.log(landmark_distances[landmark_distances != 0])

        # Logging edge success rates
        no_diagonal_edge_success_rate = edge_success_rate[~np.eye(self.num_landmarks, dtype=bool)]
        self.success_rates.append(np.mean(no_diagonal_edge_success_rate))
        self.non_zero_success_rates.append(np.mean(no_diagonal_edge_success_rate[no_diagonal_edge_success_rate > 0]))
        
        total_edges = max((np.sum(no_diagonal_edge_success_rate > self.success_threshold)) // 2, 0)
        zero_edges = 0
        self.zero_edge_indices = set()

        # Augment G with edges until it is connected
        # Edges are sorted by success rate, then similarity
        G = nx.from_numpy_array(landmark_distances)
        if not nx.is_connected(G):
            avail = []
            for index, x in np.ndenumerate(similarities):
                if non_edges[index]:
                    avail.append((edge_success_rate[index], x, *index))
            avail = sorted(avail, key=itemgetter(0, 1), reverse=True)

            for success_rate, similarity, u, v in avail:
                dist = -1 * np.log(success_rate)
                if success_rate == 0:
                    dist = -1 * np.log(zero_success_dist * similarity)
                
                landmark_distances[(u, v)] = dist
                
                G = nx.from_numpy_array(landmark_distances)

                total_edges += 1
                if success_rate == 0:
                    zero_edges += 1
                    self.zero_edge_indices.add((u, v))

                if nx.is_connected(G):
                    break
        
        self.zero_success_edge_ratio.append(zero_edges / max(total_edges, 1))

        self.graph = G
        return self.graph

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
        paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, source, target, weight='weight'), self.landmark_paths))
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
        self.paths[enter_idxs] = -1
        self.path_idxs[enter_idxs] = 0
        self.last_landmarks[enter_idxs] = -1

        self.entered_landmark_mode[enter_idxs] = True

    def set_paths(self, dsr, position):
        if not np.any(self.entered_landmark_mode):
            return
        
        norm_dsr = dsr[self.entered_landmark_mode]
        selected_position = position[self.entered_landmark_mode]

        norm_dsr = norm_dsr.mean(dim=1) / torch.norm(norm_dsr.mean(dim=1), p=2, keepdim=True)
        landmark_similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        # Select start landmarks based on SF similarity w.r.t. current observations
        start_landmarks = landmark_similarity.argmax(axis=0).cpu().detach().numpy()

        if self.mode == 'eval':
            goal_landmarks = np.full(sum(self.entered_landmark_mode), 0, dtype=int)
        else:
            # Select goal landmarks with probability given by inverse of visitation count
            inverse_visitations = 1. / np.clip(self.visitations, 1, None)
            landmark_probabilities = inverse_visitations / inverse_visitations.sum()
            goal_landmarks = np.random.choice(range(len(landmark_probabilities)),
                                            size=len(start_landmarks),
                                            replace=True,
                                            p=landmark_probabilities)

        enter_idxs = np.arange(self.num_envs)[self.entered_landmark_mode]

        for enter_idx, start_pos, start_landmark, goal_landmark in zip(enter_idxs, selected_position, start_landmarks, goal_landmarks):
            self.start_positions[enter_idx] = start_pos
            cur_x, cur_y = start_pos

            # Find correct start landmark based on true distances
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
            self.paths[enter_idx][:path_length] = path
            self.path_lengths[enter_idx] = path_length

            # Log that we attempted to reach each landmark in the path
            if self.mode != 'eval':
                self.landmark_attempts[:path_length] += 1
            
            self.entered_landmark_mode[enter_idx] = False

    def get_landmarks_data(self, current_observation, current_position):
        if not np.any(self.landmark_mode):
            return None, self.landmark_mode

        # Check if current landmark is reached based on similarity in SF space
        # norm_dsr = current_dsr[self.landmark_mode]
        # norm_dsr = norm_dsr.mean(dim=1) / torch.norm(norm_dsr.mean(dim=1), p=2, keepdim=True)

        current_idxs = self.path_idxs[self.landmark_mode]
        current_landmarks = self.paths[self.landmark_mode, current_idxs]
        # landmark_similarity = torch.matmul(self.norm_dsr[current_landmarks], norm_dsr.T)

        # TODO: Logging
        # # Log if current landmark is reached based on similarity
        # if landmark_similarity > self.reach_threshold and self._mode != 'eval':
        #     self.landmark_reaches[self.path_idx] += 1

        # reached_landmarks = landmark_similarity > self.reach_threshold
        final_goal_landmarks = current_idxs == (self.path_lengths[self.landmark_mode] - 1)
        # reached_landmarks[final_goal_landmarks] = landmark_similarity >= 1.0

        reached_landmarks = self.landmark_mode[self.landmark_mode]

        for i, observation in enumerate(current_observation[self.landmark_mode]):
            reached_landmarks[i] = torch.allclose(observation, self.observations[current_landmarks[i]])

        # Increment the current landmark's visitation count
        self.visitations[current_landmarks[reached_landmarks]] += 1

        steps_limit_reached = self.landmark_steps[self.landmark_mode] >= (self.path_lengths[self.landmark_mode] * self.steps_per_landmark)
        
        if self.mode != 'eval':
            # In training, log landmarks are truly reached 
            reaches = np.bincount(current_landmarks[reached_landmarks])
            self.landmark_true_reaches[:len(reaches)] += reaches

            # Update landmark transition success rates
            last_landmarks = self.last_landmarks[self.landmark_mode]
            reached_last_landmarks = last_landmarks != -1

            success_last_landmarks = last_landmarks[reached_last_landmarks & reached_landmarks]
            success_current_landmarks = current_landmarks[reached_last_landmarks & reached_landmarks]
            self.successes[success_last_landmarks, success_current_landmarks] += 1
            self.successes[success_current_landmarks, success_last_landmarks] += 1
            self.attempts[success_last_landmarks, success_current_landmarks] += 1
            self.attempts[success_current_landmarks, success_last_landmarks] += 1

            fail_last_landmarks = last_landmarks[reached_last_landmarks & ~reached_landmarks & steps_limit_reached]
            fail_current_landmarks = current_landmarks[reached_last_landmarks & ~reached_landmarks & steps_limit_reached]
            self.attempts[fail_last_landmarks, fail_current_landmarks] += 1
            self.attempts[fail_current_landmarks, fail_last_landmarks] += 1

        # If reached (not goal) landmark, move to next landmark
        reached_non_goal_landmarks = reached_landmarks & ~final_goal_landmarks
        self.last_landmarks[self.landmark_mode][reached_non_goal_landmarks] = current_landmarks[reached_non_goal_landmarks]
        self.path_idxs[self.landmark_mode][reached_non_goal_landmarks] += 1

        # If reached goal landmark or steps limit, exit landmark mode
        reached_goal_landmarks = reached_landmarks & final_goal_landmarks 
        end_positions = current_position[self.landmark_mode][reached_goal_landmarks | steps_limit_reached]
        end_landmarks = current_landmarks[reached_goal_landmarks | steps_limit_reached]
        goal_positions = self.positions[end_landmarks]
        end_distance = self.oracle_distance_matrix[end_positions[:, 0], end_positions[:, 1], goal_positions[:, 0], goal_positions[:, 1]]

        if self.mode == 'eval':
            # In eval, log end position trying to reach goal and distance away from goal
            for end_position, end_landmark in zip(end_positions, end_landmarks):
                self.eval_end_pos[tuple(end_position)] = end_landmark
            self.eval_distances.extend(end_distance.tolist())
        else:
            # In train, log end/start distance to goal ratio
            start_positions = self.start_positions[self.landmark_mode][reached_goal_landmarks | steps_limit_reached]
            start_distance = self.oracle_distance_matrix[start_positions[:, 0], start_positions[:, 1], goal_positions[:, 0], goal_positions[:, 1]]
            dist_completed = end_distance / np.clip(start_distance, a_min=1, a_max=None)
            self.goal_landmark_dist_completed.extend(dist_completed.tolist())

        self.landmark_mode[np.where(self.landmark_mode)[0][reached_goal_landmarks | steps_limit_reached]] = False

        # Increment landmark step counter
        self.landmark_steps[self.landmark_mode] += 1

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
        