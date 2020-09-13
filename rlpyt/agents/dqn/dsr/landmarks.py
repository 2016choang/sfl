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
                 memory_len=1,
                 landmarks_per_update=None,
                 landmark_mode_interval=100,
                 steps_per_landmark=10,
                 max_landmark_mode_steps=500,
                 localization_threshold=0.9,
                 random_true_edges_threshold=50,
                 subgoal_true_edges_threshold=10,
                 subgoal_success_threshold=10,
                 subgoal_success_true_edges_threshold=-1,
                 SF_similarity_true_edges_threshold=-1,
                 use_digraph=True,
                 use_weighted_edges=False,
                 max_attempt_threshold=1,
                 attempt_percentile_threshold=5,
                 sim_percentile_threshold=None,
                 GT_localization=False,
                 GT_localization_distance_threshold=50,
                 GT_localization_angle_threshold=30,
                 GT_termination=False,
                 GT_termination_distance_threshold=50,
                 GT_termination_angle_threshold=30,
                 GT_graph=False,
                 GT_graph_edge_threshold=100,
                 landmark_paths=1,
                 reach_threshold=0.99,
                 affinity_decay=0.9,
                ):
        save__init__args(locals())
        self.current_num_landmarks = 0
        self.num_landmarks = 0
        self.observations = None 
        self.features = None
        self.norm_dsr = None
        self.positions = None

        self.potential_landmarks = {}
        if top_k_similar is None:
            self.top_k_similar = max(int(max_landmarks * 0.10), 1)

        if landmarks_per_update is None:
            self.landmarks_per_update = max(int(max_landmarks * 0.10), 1)

        self.predecessors = None
        self.graph = None

        self.edge_random_steps = None
        self.edge_random_transitions = None
        self.edge_subgoal_steps = None
        self.edge_subgoal_successes = None
        self.edge_subgoal_transitions = None

        self.zero_edge_indices = None

        self._active = False

        self.current_sim_threshold = 0
        self.consecutive_graph_generation_successes = 0

        self.oracle_distance_matrix = None

        # DEBUGGING METRICS
        self.closest_landmarks = np.zeros(self.max_landmarks)
        self.closest_landmarks_sim = np.zeros(self.max_landmarks)
        self.transitions = 0
    
    def initialize(self, num_envs, feature_size, device, mode='train'):
        self.num_envs = num_envs
        self.feature_size = feature_size
        self.device = device
        self.mode = mode

        # Landmark mode metadata
        self.landmark_mode = np.full(self.num_envs, mode == 'eval', dtype=bool)
        self.explore_steps = np.full(self.num_envs, 0, dtype=int)
        self.landmark_steps = np.full(self.num_envs, 0, dtype=int)
        self.current_landmark_steps = np.full(self.num_envs, 0, dtype=int)
        self.paths = np.full((self.num_envs, self.max_landmarks), -1, dtype=int)
        self.path_lengths = np.full(self.num_envs, -1, dtype=int)
        self.path_idxs = np.full(self.num_envs, -1, dtype=int)
        self.start_positions = np.full((self.num_envs, 3), -1, dtype=int)
        self.entered_landmark_mode = np.full(self.num_envs, False, dtype=bool)
        
        self.last_landmarks = np.full(self.num_envs, -1, dtype=int)
        self.transition_random_steps = np.full(self.num_envs, 0, dtype=int)
        self.transition_subgoal_steps = np.full(self.num_envs, 0, dtype=int)

        self.found_eval_path = False
        self.eval_end_pos = {}
        self.eval_distances = []

        self.feature_memory = torch.zeros((self.memory_len, self.num_envs, self.feature_size), device=device)
        self.dsr_memory = torch.zeros((self.memory_len, self.num_envs, self.feature_size), device=device)
        self.memory_capacity = np.full(self.num_envs, 0, dtype=int)

        self.reset_logging()
    
    def reset(self, env_idx=None):
        if env_idx:
            self.last_landmarks[env_idx] = -1
            self.transition_random_steps[env_idx] = 0
            self.transition_subgoal_steps[env_idx] = 0
            self.memory_capacity[env_idx] = 0
        else:
            self.last_landmarks.fill(-1)
            self.transition_random_steps.fill(0)
            self.transition_subgoal_steps.fill(0)
            self.memory_capacity.fill(0)

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

        # Distance / angle diff at algo-determined localization
        self.dist_at_localization = []
        self.angle_diff_at_localization = []
        self.wall_intersections_at_localization = 0
        self.correct_localizations = 0
        self.attempted_localizations = 0

        # Distance / angle diff at algo-determined termination
        self.dist_at_termination = []
        self.angle_diff_at_termination = []
        self.wall_intersections_at_termination = 0
        self.correct_terminations = 0
        self.attempted_terminations = 0

        # Attempts used to generate fully connected landmark graph
        self.generate_graph_attempts = [] 

        # Metrics about connected components of graph
        self.graph_components = []
        self.graph_size_largest_component = []

        # Percentage of times we reach the ith landmark
        self.landmark_attempts = np.zeros(self.max_landmarks)
        # self.landmark_reaches = np.zeros(self.max_landmarks)
        self.landmark_true_reaches = np.zeros(self.max_landmarks)

        # End / start distance to goal landmark
        self.goal_landmark_dist_completed = []

        self.high_sim_positions = np.zeros((1, 6))

    def save(self, filename):
        # Save landmarks data to a file
        np.savez(filename,
                # observations=self.observations.cpu().detach().numpy(),
                features=self.features.cpu().detach().numpy(),
                norm_dsr=self.norm_dsr.cpu().detach().numpy(),
                positions=self.positions,
                edge_random_steps=self.edge_random_steps,
                edge_random_transitions=self.edge_random_transitions,
                edge_subgoal_steps=self.edge_subgoal_steps,
                edge_subgoal_successes=self.edge_subgoal_successes,
                edge_subgoal_transitions=self.edge_subgoal_transitions,
                closest_landmarks=self.closest_landmarks,
                closest_landmarks_sim=self.closest_landmarks_sim,
                high_sim_positions=self.high_sim_positions[1:])

    def update(self):
        # Decay empirical transition data by affinity_decay factor
        # self.transition_distances = (self.transition_distances * self.affinity_decay)
        # self.attempts = (self.attempts * self.affinity_decay)
        pass

        # self.interval_transition_distances = (self.interval_transition_distances * self.affinity_decay)
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
        
        # self.localizations = self.localizations[save_idx]
        # self.transition_distances = self.transition_distances[save_idx[:, None], save_idx]
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
            self.edge_random_steps = np.array([[0]])
            self.edge_random_transitions = np.array([[0]])
            self.edge_subgoal_steps = np.array([[0]])
            self.edge_subgoal_successes = np.array([[0]])
            self.edge_subgoal_transitions = np.array([[0]])
            self.num_landmarks += 1
        else:
            # self.observations = torch.cat((self.observations, observation), dim=0)
            self.set_features(features, self.num_landmarks)
            self.set_dsr(dsr, self.num_landmarks)
            self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            

            self.edge_random_steps = np.append(self.edge_random_steps, np.zeros((self.num_landmarks, 1)), axis=1)
            self.edge_random_steps = np.append(self.edge_random_steps, np.zeros((1, self.num_landmarks + 1)), axis=0)

            self.edge_random_transitions = np.append(self.edge_random_transitions, np.zeros((self.num_landmarks, 1)), axis=1)
            self.edge_random_transitions = np.append(self.edge_random_transitions, np.zeros((1, self.num_landmarks + 1)), axis=0)

            self.edge_subgoal_steps = np.append(self.edge_subgoal_steps, np.zeros((self.num_landmarks, 1)), axis=1)
            self.edge_subgoal_steps = np.append(self.edge_subgoal_steps, np.zeros((1, self.num_landmarks + 1)), axis=0)

            self.edge_subgoal_successes = np.append(self.edge_subgoal_successes, np.zeros((self.num_landmarks, 1)), axis=1)
            self.edge_subgoal_successes = np.append(self.edge_subgoal_successes, np.zeros((1, self.num_landmarks + 1)), axis=0)

            self.edge_subgoal_transitions = np.append(self.edge_subgoal_transitions, np.zeros((self.num_landmarks, 1)), axis=1)
            self.edge_subgoal_transitions = np.append(self.edge_subgoal_transitions, np.zeros((1, self.num_landmarks + 1)), axis=0)

            self.num_landmarks += 1

    def force_remove_landmark(self):
        # Remove last landmark
        save_idx = range(self.num_landmarks - 1)

        self.features = self.features[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]

        self.positions = self.positions[save_idx]

        self.num_landmarks -= 1
        self.edge_random_steps = self.edge_random_steps[:self.num_landmarks, :self.num_landmarks]
        self.edge_random_transitions = self.edge_random_transitions[:self.num_landmarks, :self.num_landmarks]
        self.edge_subgoal_steps = self.edge_subgoal_steps[:self.num_landmarks, :self.num_landmarks]
        self.edge_subgoal_successes = self.edge_subgoal_successes[:self.num_landmarks, :self.num_landmarks]
        self.edge_subgoal_transitions = self.edge_subgoal_transitions[:self.num_landmarks, :self.num_landmarks]
    
    def update_memory(self, feature, norm_dsr):
        self.feature_memory = torch.roll(self.feature_memory, -1, dims=0)
        self.feature_memory[-1] = feature
        self.dsr_memory = torch.roll(self.dsr_memory, -1, dims=0)
        self.dsr_memory[-1] = norm_dsr
        self.memory_capacity[self.memory_capacity < self.memory_len] += 1
    
    @property
    def current_dsr(self):
        return self.norm_dsr[:self.current_num_landmarks]

    @property
    def current_positions(self):
        return self.positions[:self.current_num_landmarks]

    def analyze_current_state(self, features, dsr, position):
        if self.num_landmarks > 0:
            mean_dsr = dsr.mean(dim=1)
            norm_dsr = mean_dsr / torch.norm(mean_dsr, p=2, dim=1, keepdim=True)
            self.update_memory(features, norm_dsr)

        if self.current_num_landmarks > 0:
            #  DSR Memory: M x E x F
            #  Landmarks DSR: L x M x F
            self.memory_similarity = torch.bmm(self.dsr_memory, self.current_dsr.permute(1, 2, 0)).cpu().numpy()
            self.median_similarity = np.median(self.memory_similarity, axis=0) # E x L
            if self.mode == 'eval':
                return

            # Localization
            # localized_envs = torch.any(similarity >= self.localization_threshold, dim=0).cpu().numpy()  # localized to some landmark
            # closest_landmarks = torch.argmax(similarity, dim=0).cpu().numpy()  # get landmarks with highest similarity to current state 
            # closest_landmarks_sim = torch.max(similarity, dim=0)[0].cpu().numpy()
            
            localized_envs = (self.memory_capacity == self.memory_len) & np.any(self.median_similarity >= self.localization_threshold, axis=1)  # localized to some landmark
            closest_landmarks = np.argmax(self.median_similarity, axis=1)  # get landmarks with highest similarity to current state 
            closest_landmarks_sim = np.max(self.median_similarity, axis=1)

            self.closest_landmarks[closest_landmarks[localized_envs]] += 1
            self.closest_landmarks_sim[closest_landmarks[localized_envs]] += closest_landmarks_sim[localized_envs]

            for pos, landmark in zip(position[localized_envs], closest_landmarks[localized_envs]):
                distance, intersection = self.get_oracle_distance_to_landmarks(pos, [landmark])
                angle_diff = abs(pos[2] - self.current_positions[landmark, 2])
                if not np.any(intersection):
                    self.dist_at_localization.append(distance[0])
                    if distance < self.GT_localization_distance_threshold and angle_diff < self.GT_localization_angle_threshold:
                        self.correct_localizations += 1
                    # else:
                    #    self.high_sim_positions = np.append(self.high_sim_positions, np.concatenate([pos, self.positions[landmark]])[np.newaxis], axis=0)
                else:
                    self.wall_intersections_at_localization += np.sum(intersection)
                    self.high_sim_positions = np.append(self.high_sim_positions, np.concatenate([pos, self.positions[landmark]])[np.newaxis], axis=0)
                    
                self.angle_diff_at_localization.append(angle_diff)

            self.attempted_localizations += np.sum(localized_envs)

            if self.GT_localization:
                GT_distance = np.column_stack([self.get_oracle_distance_to_landmarks(pos, intersection_penalty=True)[0] for pos in position])  # L x E
                GT_angle = np.abs(position[np.newaxis, :, 2] - self.current_positions[:, np.newaxis, 2])  # L x E
                localized_envs = np.any((GT_distance < self.GT_localization_distance_threshold) & (GT_angle < self.GT_localization_angle_threshold), axis=0)
                closest_landmarks = np.argmin(GT_distance, axis=0)

            new_localizations = localized_envs & (self.last_landmarks != closest_landmarks)  # localized to some new landmark

            transitions = (self.last_landmarks != -1) & new_localizations  # transitioned between landmarks
            self.transitions += np.sum(transitions)

            transition_last_landmarks = self.last_landmarks[transitions]
            transition_next_landmarks = closest_landmarks[transitions]

            random_steps = self.transition_random_steps[transitions]
            subgoal_steps = self.transition_subgoal_steps[transitions]

            current_subgoal_landmarks = self.paths[np.arange(len(self.path_idxs)), self.path_idxs]
            subgoal_localized_wrong_landmark = (self.landmark_mode & (current_subgoal_landmarks != closest_landmarks))[transitions]

            random_transitions = (random_steps > 0) | subgoal_localized_wrong_landmark

            self.edge_random_steps[transition_last_landmarks, transition_next_landmarks] += ((random_steps + subgoal_steps) * (random_transitions))
            self.edge_random_transitions[transition_last_landmarks, transition_next_landmarks] += (random_transitions)

            self.edge_subgoal_steps[transition_last_landmarks, transition_next_landmarks] += (subgoal_steps * ~random_transitions)
            self.edge_subgoal_successes[transition_last_landmarks, transition_next_landmarks] += (~random_transitions & (subgoal_steps < self.subgoal_success_threshold))

            self.edge_subgoal_transitions[transition_last_landmarks, transition_next_landmarks] += (~random_transitions)
            
            self.last_landmarks[new_localizations] = closest_landmarks[new_localizations]
            self.transition_random_steps[new_localizations] = 0
            self.transition_subgoal_steps[new_localizations] = 0
        
        for idx in range(self.num_envs):
            self.add_landmark(idx, position[idx], self.last_landmarks[idx], self.transition_random_steps[idx], self.transition_subgoal_steps[idx])

    def add_landmark(self, idx, position, last_landmark=-1, random_steps=None, subgoal_steps=None):
        # Add landmark if it is not similar w.r.t. existing landmarks
        features = self.feature_memory[:, idx].unsqueeze(0)
        dsr = self.dsr_memory[:, idx].unsqueeze(0)
        if self.num_landmarks == 0:
            # First landmark
            # self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr)
            self.positions = np.expand_dims(position, 0)
            self.edge_random_steps = np.array([[0]])
            self.edge_random_transitions = np.array([[0]])
            self.edge_subgoal_steps = np.array([[0]])
            self.edge_subgoal_successes = np.array([[0]])
            self.edge_subgoal_transitions = np.array([[0]])
            self.landmark_adds += 1
            self.num_landmarks += 1
            return True
        else:
            similarity = torch.matmul(self.norm_dsr[:, -1], self.dsr_memory[-1, idx])  # Compute similarity w.r.t. each existing landmark |num landmarks|

            # Candidate landmark under similarity threshold w.r.t. existing landmarks
            if torch.sum(similarity < self.add_threshold) >= self.num_landmarks:
                self.landmark_adds += 1

                # Add landmark
                if self.num_landmarks < self.max_landmarks:
                    # self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks)
                    self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            

                    self.edge_random_steps = np.append(self.edge_random_steps, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.edge_random_steps = np.append(self.edge_random_steps, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.edge_random_transitions = np.append(self.edge_random_transitions, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.edge_random_transitions = np.append(self.edge_random_transitions, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.edge_subgoal_steps = np.append(self.edge_subgoal_steps, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.edge_subgoal_steps = np.append(self.edge_subgoal_steps, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.edge_subgoal_successes = np.append(self.edge_subgoal_successes, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.edge_subgoal_successes = np.append(self.edge_subgoal_successes, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.edge_subgoal_transitions = np.append(self.edge_subgoal_transitions, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.edge_subgoal_transitions = np.append(self.edge_subgoal_transitions, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    self.landmark_adds += 1
                    self.num_landmarks += 1

                    new_landmark = self.num_landmarks - 1

                # Replace existing landmark
                else:
                    visitations = np.sum(self.edge_random_transitions, axis=0) + np.sum(self.edge_subgoal_transitions, axis=0)
                    replace_idx = np.argmin(visitations)

                    # self.observations[replace_idx] = observation
                    self.set_features(features, replace_idx)
                    self.set_dsr(dsr, replace_idx)
                    self.positions[replace_idx] = np.expand_dims(position, 0)

                    self.edge_random_steps[replace_idx, :] = 0
                    self.edge_random_steps[:, replace_idx] = 0
                    self.edge_random_transitions[replace_idx, :] = 0
                    self.edge_random_transitions[replace_idx, :] = 0
                    self.edge_subgoal_steps[replace_idx, :] = 0
                    self.edge_subgoal_steps[:, replace_idx] = 0
                    self.edge_subgoal_successes[:, replace_idx] = 0
                    self.edge_subgoal_successes[:, replace_idx] = 0
                    self.edge_subgoal_transitions[:, replace_idx] = 0
                    self.edge_subgoal_transitions[:, replace_idx] = 0

                    self.landmark_removes += 1

                    new_landmark = replace_idx
                    
                    replaced_landmarks = self.last_landmarks == replace_idx
                    self.last_landmarks[replaced_landmarks] = -1
                    self.transition_random_steps[replaced_landmarks] = 0
                    self.transition_random_steps[replaced_landmarks] = 0
                
                if last_landmark != -1:
                    
                    if random_steps > 0:
                        self.edge_random_steps[last_landmark, new_landmark] += (random_steps + subgoal_steps)
                        self.edge_random_transitions[last_landmark, new_landmark] += 1
                    
                    else:
                        self.edge_subgoal_steps[last_landmark, new_landmark] += subgoal_steps
                        self.edge_subgoal_successes[last_landmark, new_landmark] += (subgoal_steps < self.subgoal_success_threshold)
                        self.edge_subgoal_transitions[last_landmark, new_landmark] += 1
                        
                return True

            else:
                return False
            
    def set_features(self, features, idx=None):
        # Set/add features of new landmark at idx
        if self.features is None or idx is None:
            self.features = features
        elif isinstance(idx, np.ndarray) or (0 <= idx and idx < self.num_landmarks):
            self.features[idx] = features
        else:
            self.features = torch.cat((self.features, features), dim=0)

    def set_dsr(self, norm_dsr, idx=None):
        # Set/add DSR of new landmark at idx 
        if self.norm_dsr is None or idx is None:
            self.norm_dsr = norm_dsr
        elif isinstance(idx, np.ndarray) or (0 <= idx and idx < self.num_landmarks):
            self.norm_dsr[idx] = norm_dsr
        else:
            self.norm_dsr = torch.cat((self.norm_dsr, norm_dsr), dim=0)

    def generate_graph(self):
        if self.GT_graph:
            edges = np.array(list(itertools.product(self.positions[:, :2], self.positions[:, :2]))).reshape(-1, 4)
            self.landmark_distances = np.linalg.norm(edges[:, :2] - edges[:, 2:], ord=2, axis=1)
            
            intersections = self.get_intersections(edges.T)
            self.landmark_distances[intersections] += (self.max_landmarks * self.landmark_distances.max())
            self.landmark_distances[self.landmark_distances == 0] += 1e-3
            self.landmark_distances = self.landmark_distances.reshape((self.num_landmarks, self.num_landmarks))

            over_edge_threshold = self.landmark_distances > self.GT_graph_edge_threshold

            graph_distance_matrix = self.landmark_distances.copy()
            graph_distance_matrix[over_edge_threshold] = 0

            self.graph = nx.from_numpy_array(graph_distance_matrix, create_using=nx.DiGraph)
            if nx.is_strongly_connected(self.graph):
                return self.graph
            
            under_twice_edge_threshold = self.landmark_distances <= (2 * self.GT_graph_edge_threshold)
            graph_distance_matrix[over_edge_threshold & under_twice_edge_threshold] = (self.max_landmarks * self.GT_graph_edge_threshold) \
                + self.landmark_distances[over_edge_threshold & under_twice_edge_threshold]
            self.graph = nx.from_numpy_array(graph_distance_matrix, create_using=nx.DiGraph)
            if nx.is_strongly_connected(self.graph):
                return self.graph

            self.landmark_distances[~under_twice_edge_threshold] += (self.max_landmarks * 2 * self.GT_graph_edge_threshold)
            self.graph = nx.from_numpy_array(self.landmark_distances, create_using=nx.DiGraph)
            if not nx.is_strongly_connected(self.graph):
                raise RuntimeError('Graph should be complete and therefore, strongly connected')
            return self.graph
        else:
            if self.use_digraph:
                random_steps = self.edge_random_steps
                random_transitions = self.edge_random_transitions

                subgoal_steps = self.edge_subgoal_steps
                subgoal_successes = self.edge_subgoal_successes
                subgoal_transitions = self.edge_subgoal_transitions

            else:
                random_steps = self.edge_random_steps + np.tril(self.edge_random_steps, -1).T
                random_transitions = self.edge_random_transitions + np.tril(self.edge_random_transitions, -1).T
            
                subgoal_steps = self.edge_subgoal_steps + np.tril(self.edge_subgoal_steps, -1).T
                subgoal_successes = self.edge_subgoal_successes + np.tril(self.edge_subgoal_successes, -1).T
                subgoal_transitions = self.edge_subgoal_transitions + np.tril(self.edge_subgoal_transitions, -1).T

            average_random_steps = random_steps / np.clip(random_transitions, 1, None)
            average_subgoal_steps = subgoal_steps / np.clip(subgoal_transitions, 1, None)

            true_edges = ((average_random_steps < self.random_true_edges_threshold) & (random_transitions > 0)) | \
                ((average_subgoal_steps < self.subgoal_true_edges_threshold) & (subgoal_transitions > 0))
            
            if self.subgoal_success_true_edges_threshold != -1:
                percentage_subgoal_successes = subgoal_successes / np.clip(subgoal_transitions, 1, None)
                true_edges |= (percentage_subgoal_successes > self.subgoal_success_true_edges_threshold)
            
            if self.SF_similarity_true_edges_threshold != -1:
                norm_dsr = self.norm_dsr[:, -1]  # DSR of last frame
                SF_similarity = torch.matmul(norm_dsr, norm_dsr.T).cpu().numpy()
                true_edges &= (SF_similarity > self.SF_similarity_true_edges_threshold)


            # feature_similarity = torch.clamp(torch.matmul(self.norm_dsr, self.norm_dsr.T), min=1e-3, max=1.0).detach().cpu().numpy()
            # true_edges &= (feature_similarity > self.graph_feature_similarity_threshold)

            if self.use_weighted_edges:
                edge_weights = true_edges * ((0.5 * average_random_steps + average_subgoal_steps) / np.clip(0.5 * (random_transitions > 0) + (subgoal_transitions > 0), 1, None))
                if self.subgoal_success_true_edges_threshold != -1:
                    edge_weights[(percentage_subgoal_successes <= self.subgoal_success_true_edges_threshold) & (true_edges)] += (edge_weights.max() * self.max_landmarks)
            else:
                edge_weights = true_edges

            self.landmark_distances = edge_weights

            self.graph = nx.from_numpy_array(self.landmark_distances, create_using=nx.DiGraph)
            
            self.graph_components.append(nx.number_strongly_connected_components(self.graph))
            self.graph_size_largest_component.append(len(max(nx.strongly_connected_components(self.graph), key=len)))

            return self.graph

    def connect_goal(self):
        if self.num_landmarks > 1:
            self.landmark_distances = np.append(self.landmark_distances, np.zeros((self.num_landmarks - 1, 1)), axis=1)
            self.landmark_distances = np.append(self.landmark_distances, np.zeros((1, self.num_landmarks)), axis=0)

            if self.GT_graph:
                goal_pos = self.positions[-1, :2]
                oracle_distance_to_goal = self.get_oracle_distance_to_landmarks(goal_pos)
                closest_to_goal = oracle_distance_to_goal[:-1].argmin()
            else:
                similarity_to_goal = (self.norm_dsr[-1, -1] * self.norm_dsr[:-1, -1]).sum(dim=1)
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
    
    def get_intersections(self, edges):
        def orientation(p, q, r):
            return ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])) 
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0])) & (q[0] > min(p[0], r[0])) & (q[1] <= max(p[1], r[1])) & (q[1] >= min(p[1], r[1]))

        orientations = np.zeros((edges.shape[1], len(self.lines), 4))
        for i, line in enumerate(self.lines):
            orientations[:, i, 0] = orientation(edges[:2], edges[2:], line[:2])
            orientations[:, i, 1] = orientation(edges[:2], edges[2:], line[2:])
            orientations[:, i, 2] = orientation(line[:2], line[2:], edges[:2])
            orientations[:, i, 3] = orientation(line[:2], line[2:], edges[2:])

        orientations[orientations > 0] = 1
        orientations[orientations < 0] = 2

        intersections = np.zeros((edges.shape[1], len(self.lines)), dtype=bool)
        intersections[(orientations[:, :, 0] != orientations[:, :, 1]) & (orientations[:, :, 2] != orientations[:, :, 3])] = True

        for i, j, k in np.argwhere(orientations == 0):
            if k == 0:
                p, q, r = edges[:2, i], self.lines[j, :2], edges[2:, i] 
            elif k == 1:
                p, q, r = edges[:2, i], self.lines[j, 2:], edges[2:, i]
            elif k == 2:
                p, q, r = self.lines[j, :2], edges[:2, i], self.lines[j, 2:]
            else:
                p, q, r = self.lines[j, :2], edges[2:, i], self.lines[j, 2:]
            intersections[i, j] = on_segment(p, q, r)
        return np.any(intersections, axis=1)

    def get_oracle_distance_to_landmarks(self, pos, idxs=None, intersection_penalty=False):
        if idxs:
            positions = self.positions[idxs, :2]
        else:
            positions = self.current_positions[:, :2]
        distance = np.linalg.norm(positions - pos[:2], ord=2, axis=1)

        broadcasted_pos = np.broadcast_to(pos[np.newaxis, :2], positions.shape)
        edges = np.hstack((positions, broadcasted_pos)).T
        intersections = self.get_intersections(edges)
        if intersection_penalty:
            distance[intersections] += (distance.max() * self.max_landmarks)
        return distance, intersections
        
    def set_paths(self, position, relocalize_idxs=None):
        if relocalize_idxs is None:
            set_paths_idxs = self.entered_landmark_mode.copy()
            self.entered_landmark_mode[self.entered_landmark_mode] = False
        else:
            set_paths_idxs = relocalize_idxs

        if not np.any(set_paths_idxs):
            return
        
        selected_position = position[set_paths_idxs]

        landmark_similarity = self.memory_similarity[-1, set_paths_idxs]

        # Select start landmarks based on SF similarity w.r.t. current observations
        start_landmarks = landmark_similarity.argmax(axis=0).cpu().detach().numpy()
        if relocalize_idxs is None:
            if self.mode == 'eval':
                # Goal landmark is set as the last landmark to be added
                goal_landmarks = np.full(sum(set_paths_idxs), self.num_landmarks - 1, dtype=int)
            else:
                # Select goal landmarks with probability given by inverse of visitation count
                # visitations = np.clip(self.localizations, 1, None)
                goal_landmarks = None
        else:
            prev_start_landmarks = self.paths[relocalize_idxs, 0]
            goal_landmarks = self.paths[relocalize_idxs, self.path_lengths[relocalize_idxs] - 1]
            self.current_landmark_steps[relocalize_idxs] = 0
            self.paths[relocalize_idxs, :] = -1
            self.path_idxs[relocalize_idxs, :] = 0
            self.last_landmarks[relocalize_idxs] = -1

        enter_idxs = np.arange(self.num_envs)[set_paths_idxs]

        if goal_landmarks is None:
            components = sorted(nx.strongly_connected_components(self.graph), key=len, reverse=True)
            visitations = np.clip(np.sum(self.edge_random_transitions, axis=0) + np.sum(self.edge_subgoal_transitions, axis=0), 1, None)[:self.current_num_landmarks]
            inverse_visitations = 1. / visitations

        for i, enter_idx, start_pos, start_landmark in zip(range(len(enter_idxs)), enter_idxs, selected_position, start_landmarks):
            if relocalize_idxs is not None and start_landmark == prev_start_landmarks[i]:
                continue

            self.start_positions[enter_idx] = start_pos
            cur_x, cur_y, cur_angle = start_pos

            oracle_distance_to_landmarks, intersections = self.get_oracle_distance_to_landmarks(start_pos[:2])
            dist_to_selected_start = oracle_distance_to_landmarks[start_landmark]
            # Intersection penalty
            if intersections[start_landmark]:
                dist_to_selected_start *= 2
            oracle_distance_to_landmarks[intersections] += (self.max_landmarks * oracle_distance_to_landmarks.max())
            dist_to_estimated_best_start = oracle_distance_to_landmarks.min()

            # Find correct start landmark based on true distances
            if self.GT_localization:
                start_landmark = oracle_distance_to_landmarks.argmin()

            self.dist_start_landmark.append(dist_to_selected_start)
            self.dist_ratio_start_landmark.append(dist_to_selected_start / np.clip(dist_to_estimated_best_start, 1, None))

            if goal_landmarks is None:
                for component in components:
                    if start_landmark in component:
                        component = list(component)
                        current_inverse_visitations = inverse_visitations[component]
                        landmark_probabilities = current_inverse_visitations / current_inverse_visitations.sum()
                        goal_idx = np.random.choice(range(len(landmark_probabilities)), p=landmark_probabilities)
                        goal_landmark = component[goal_idx] 
                        break
                        
            else:
                goal_landmark = goal_landmarks[i]
                if self.mode == 'eval':
                    if not nx.has_path(self.graph, start_landmark, goal_landmark):
                        self.landmark_mode[enter_idx] = False
                        continue
                    else:
                        self.found_eval_path = True

            # if self.oracle_distance_matrix:
            #     closest_landmark = None
            #     min_dist = None
            #     for i, pos in enumerate(self.positions):
            #         dist = self.oracle_distance_matrix[cur_x, cur_y, pos[0], pos[1]]
            #         if min_dist is None or dist < min_dist:
            #             min_dist = dist
            #             closest_landmark = i

            #         if i == start_landmark:
            #             chosen_landmark_dist = dist

            #     # Log if selected landmark is correct
            #     self.total_landmark_paths += 1
            #     if start_landmark == closest_landmark:
            #         self.correct_start_landmark += 1

            #     # Log ratio of distance to selected landmark / correct landmark
            #     if chosen_landmark_dist:
            #         self.dist_ratio_start_landmark.append(min_dist / chosen_landmark_dist)
            #     else:
            #         self.dist_ratio_start_landmark.append(1)

            #     # ORACLE: Use correct start landmark
            #     if self.use_oracle_start:
            #         start_landmark = closest_landmark

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
        end_distance = euclidean_distance(pos[:2], goal_pos[:2])
        self.eval_distances.append(end_distance)

    def get_landmarks_data(self, current_position):
        if not np.any(self.landmark_mode) or self.current_num_landmarks == 0:
            return None, self.landmark_mode, None

        current_idxs = self.path_idxs[self.landmark_mode]
        current_landmarks = self.paths[self.landmark_mode, current_idxs]
        final_goal_landmarks = current_idxs == (self.path_lengths[self.landmark_mode] - 1)

        # Termination based on SF similarity
        # norm_dsr = current_dsr[self.landmark_mode]
        # norm_dsr = norm_dsr.mean(dim=1) / torch.norm(norm_dsr.mean(dim=1), p=2, dim=1, keepdim=True)
        # landmark_similarity = torch.sum(norm_dsr * self.norm_dsr[current_landmarks], dim=1)

        similarity = self.median_similarity[self.landmark_mode, current_landmarks]  # up to dim = |E|
        reached_landmarks = (self.memory_capacity[self.landmark_mode] == self.memory_len) & (similarity > self.reach_threshold)

        for pos, landmark in zip(current_position[np.where(self.landmark_mode)[0][reached_landmarks]],
                                 current_landmarks[reached_landmarks]):
            distance, intersection = self.get_oracle_distance_to_landmarks(pos, [landmark])
            angle_diff = abs(pos[2] - self.positions[landmark, 2])
            if not np.any(intersection):
                self.dist_at_termination.append(distance[0])
                if distance < self.GT_termination_distance_threshold and angle_diff < self.GT_termination_angle_threshold:
                    self.correct_terminations += 1
            else:
                self.wall_intersections_at_termination += np.sum(intersection)
            self.angle_diff_at_termination.append(angle_diff)

        self.attempted_terminations += np.sum(reached_landmarks)

        if self.GT_termination:
            GT_distance = np.hstack([self.get_oracle_distance_to_landmarks(pos, [current_landmark], intersection_penalty=True)[0] \
                for pos, current_landmark in zip(current_position[self.landmark_mode], current_landmarks)])
            GT_angle = np.abs(current_position[self.landmark_mode, 2] - self.positions[current_landmarks, 2])
            reached_landmarks = (GT_distance < self.GT_termination_distance_threshold) & (GT_angle < self.GT_termination_angle_threshold)

        if self.mode == 'eval':
            reached_landmarks[final_goal_landmarks] = False

        # # Localization based on observation equivalence
        # reached_landmarks = self.landmark_mode[self.landmark_mode]
        # for i, observation in enumerate(current_observation[self.landmark_mode]):
        #     reached_landmarks[i] = torch.allclose(observation, self.observations[current_landmarks[i]])

        # for i, observation in enumerate(current_observation[self.landmark_mode]):
        #     if current_landmarks[i] == 0:
        #         reached_landmarks[i] = torch.allclose(observation, self.observations[current_landmarks[i]])

        # If eval mode, keep trying to reach goal until episode terminates

        # Increment the current landmark's visitation count
        # self.localizations[current_landmarks[reached_landmarks]] += 1

        if self.mode != 'eval':
            steps_limit = np.minimum(self.path_lengths[self.landmark_mode] * self.steps_per_landmark, self.max_landmark_mode_steps)
            steps_limit_reached = self.landmark_steps[self.landmark_mode] >= steps_limit
        else:
            steps_limit_reached = False

        # reached_within_steps = self.current_landmark_steps[self.landmark_mode] < self.steps_per_landmark

        # # Relocalize agent which has failed to reach current landmark in steps_per_landmark
        # if self.mode == 'eval':
        #     relocalize_idxs = self.landmark_mode.copy()
        #     relocalize_idxs[self.landmark_mode] &= ~reached_landmarks & ~reached_within_steps & ~final_goal_landmarks
        #     self.set_paths(current_dsr, current_position, relocalize_idxs)
        
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
        return self.norm_dsr[current_landmarks,], self.landmark_mode, self.positions[current_landmarks]

    ################################## UNUSED CODE ################################## 

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
