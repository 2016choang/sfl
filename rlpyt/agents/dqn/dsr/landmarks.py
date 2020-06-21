import itertools
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.special import softmax
import torch

from rlpyt.utils.quick_args import save__init__args


class Landmarks(object):

    def __init__(self,
                 max_landmarks,
                 add_threshold=0.75,
                 top_k_similar=None,
                 landmarks_per_update=None,
                 success_threshold=0,
                 sim_threshold=0.9,
                 landmark_paths=None,
                 affinity_decay=0.9):
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
    
    def reset_logging(self):
        # Reset trackers for logging
        self.landmark_adds = 0
        self.landmark_removes = 0

        self.success_rates = []
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
        norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
        similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        # Candidate landmark under similarity threshold w.r.t. existing landmarks
        if sum(similarity < self.add_threshold) >= self.num_landmarks:
            if self.potential_landmarks:
                self.potential_landmarks['observation'] = torch.cat((self.potential_landmarks['observation'],
                                                                     observation), dim=0)
                self.potential_landmarks['positions'] = np.append(self.potential_landmarks['positions'],
                                                                  np.expand_dims(position, 0), axis=0)
            else:
                self.potential_landmarks['observation'] = observation
                self.potential_landmarks['positions'] = np.expand_dims(position, 0)
            
            self.potential_landmark_adds += 1

    def add_landmarks(self, observation, features, dsr, position):
        # Add landmarks if it is not similar w.r.t existing landmarks
        # by selecting from pool of potential landmarks
        norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
        similarity = torch.matmul(self.norm_dsr, norm_dsr.T)

        k_nearest_similarity = torch.topk(similarity, 3, dim=0, largest=False).values
        similarity_score = torch.sum(k_nearest_similarity, dim=0)
        new_landmark_idxs = torch.topk(similarity_score, 2, largest=False).indices

        for idx in new_landmark_idxs:
            self.add_landmark(observation[idx], features[idx], dsr[idx], position[idx])

        # Dynamically adjust add threshold depending on value of self.potential_landmark_adds
        if self.potential_landmark_adds > self.max_landmarks:
            self.add_threshold -= 0.05
        elif self.potential_landmark_adds == 0:
            self.add_threshold += 0.01
        
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

    def generate_true_graph(self, env_true_dist, edge_threshold=None):
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
        
    def generate_graph(self, mode):
        # Generate landmark graph using empirical transitions
        # and similarity in SF space between landmarks
        N = self.num_landmarks
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

        # If less than 5% of possible edges have non-zero success rates,
        # then use edges with high similarity 
        # if sum(edge_success_rate > 0) < 0.05 * (N * (N - 1) / 2):
        #     high_similarity_edges = similarities > self.sim_threshold
        #     non_edges[high_similarity_edges] = False 
        #     landmark_distances[high_similarity_edges] = np.clip(landmark_distances[high_similarity_edges], a_min=min_dist[high_similarity_edges], a_max=None)

        # In all modes except eval, consider edges with low numbers
        # of attempted transitions as valid starting edges
        if mode != 'eval':
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

    def generate_path(self, source, target, mode):
        # Generate path from source to target in landmark graph
        
        # Get k shortest paths (k = self.landmark_paths)
        paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, source, target, weight='weight'), self.landmark_paths))
        if mode != 'eval':
            # Weights defined by index of path, where paths sorted by distance
            path_weights = np.arange(1, len(paths) + 1)
        else:
            # Weights defined by path weight
            path_weights = np.array([self.get_path_weight(path, 'weight') for path in paths])

        # Select path with probability given by softmin of path weights
        path_p = softmax(-1 * path_weights)
        path_choice = np.random.choice(list(range(len(paths))), p=path_p)
        self.path = paths[path_choice]

        if mode == 'eval':
            self.eval_paths = paths
            self.eval_paths_p = path_p
        return self.path
