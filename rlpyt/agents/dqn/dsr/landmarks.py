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
                 threshold=0.75,
                 landmark_paths=None,
                 use_affinity=True,
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
        self.actions = None

        self.predecessors = None
        self.graph = None

        self.successes = None
        self.attempts = None
        self.last_landmark = None
        self.zero_edge_indices = None

        self.reset_logging()        
    
    def reset_logging(self):
        self.landmark_adds = 0
        self.landmark_removes = 0

        self.success_rates = []
        self.non_zero_success_rates = []
        self.zero_success_edge_ratio = []

    def save(self, filename):
        np.savez(filename,
                 observations=self.observations.cpu().detach().numpy(),
                 features=self.features.cpu().detach().numpy(),
                 dsr=self.dsr.cpu().detach().numpy(),
                 visitations=self.visitations,
                 positions=self.positions,
                 actions=self.actions,
                 successes=self.successes,
                 attempts=self.attempts)

    def force_add_landmark(self, observation, features, dsr, position, action=-1):
        # Add landmark while ignoring similarity thresholds and max landmarks

        if self.num_landmarks == 0:
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr, None, action)
            self.positions = np.expand_dims(position, 0)
            self.actions = np.array([action])
            self.visitations = np.array([0])
            self.num_landmarks += 1
            self.successes = np.array([[0]])
            self.attempts = np.array([[0]])

        else:
            self.observations = torch.cat((self.observations, observation), dim=0)
            self.set_features(features, self.num_landmarks)
            self.set_dsr(dsr, self.num_landmarks, action)
            self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)
            self.actions = np.append(self.actions, action)       
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
        self.actions = self.actions[save_idx]
        self.visitations = self.visitations[save_idx]

        self.num_landmarks -= 1
        self.successes = self.successes[:self.num_landmarks, :self.num_landmarks]
        self.attempts = self.attempts[:self.num_landmarks, :self.num_landmarks]

    def add_landmark(self, observation, features, dsr, position, action=-1):
        # First landmark
        if self.num_landmarks == 0:
            self.observations = observation
            self.set_features(features)
            self.set_dsr(dsr, None)
            self.positions = np.expand_dims(position, 0)
            self.actions = np.array([action])
            self.visitations = np.array([0])
            self.successes = np.array([[0]])
            self.attempts = np.array([[0]])
            self.landmark_adds += 1
            self.last_landmark = self.num_landmarks

            self.num_landmarks += 1
        else:
            if action != -1:
                norm_dsr = dsr[:, action]
            else:
                norm_dsr = dsr.mean(dim=1)
            norm_dsr = norm_dsr / torch.norm(norm_dsr, p=2, keepdim=True)
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T)
            current_landmark = similarity.argmax().item()

            # Candidate under similarity threshold w.r.t. existing landmarks
            if sum(similarity < self.threshold) >= (self.num_landmarks - 1):
                self.landmark_adds += 1

                # Add landmark
                if self.num_landmarks < self.max_landmarks:
                    self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks, action)
                    self.positions = np.append(self.positions, np.expand_dims(position, 0), axis=0)            
                    self.actions = np.append(self.actions, action)

                    self.successes = np.append(self.successes, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.successes = np.append(self.successes, np.zeros((1, self.num_landmarks + 1)), axis=0)
                    self.attempts = np.append(self.attempts, np.zeros((self.num_landmarks, 1)), axis=1)
                    self.attempts = np.append(self.attempts, np.zeros((1, self.num_landmarks + 1)), axis=0)

                    current_landmark = self.num_landmarks
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
                    self.set_dsr(dsr, replace_idx, action)
                    self.positions[replace_idx] = np.expand_dims(position, 0)
                    self.actions[replace_idx] = action
                    self.visitations[replace_idx] = 0

                    current_landmark = replace_idx

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
            
            self.last_landmark = current_landmark

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

    def set_dsr(self, dsr, idx=None, action=-1):
        if action != -1:
            dsr = dsr[:, action]
        else:
            dsr = dsr.mean(dim=1)
        # Set/add DSR of new landmark at idx 
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
        # TODO: Hack to generate landmark graph based on true distances
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
        
    def generate_graph(self, edge_threshold, mode):
        # Generate landmark graph using (1 - similarity) between DSR of landmarks as edge weights 
        similarities = torch.clamp(torch.matmul(self.norm_dsr, self.norm_dsr.T), min=-1.0, max=1.0)
        similarities = similarities.detach().cpu().numpy()

        edge_success_rate = self.successes / np.clip(self.attempts, 1, None)
        non_zero_success = edge_success_rate[edge_success_rate > 0]
        if non_zero_success.size == 0:
            zero_success_dist = 1e-3
        else:
            zero_success_dist = non_zero_success.min()

        landmark_distances = edge_success_rate.copy()
        # landmark_distances = 1.001 - similarities
        # landmark_distances = 1.001 - edge_success_rate

        if self.use_affinity:
            # Remove edges with success rate <= edge threshold
            non_edges = np.logical_not(edge_success_rate > edge_threshold)

            if mode != 'eval':
                non_zero_attempts = self.attempts[self.attempts > 0]
                attempt_threshold = max(non_zero_attempts.mean() - non_zero_attempts.std(), 1)
                low_attempt_edges = self.attempts < attempt_threshold
                non_edges[low_attempt_edges] = False            
                landmark_distances[low_attempt_edges] = np.clip(landmark_distances[low_attempt_edges], a_min=zero_success_dist, a_max=None)
        else:
            non_edges = similarities < edge_threshold

        landmark_distances[non_edges] = 0
        landmark_distances[landmark_distances == 1] = 1 - 1e-6
        landmark_distances[landmark_distances != 0] = -1 * np.log(landmark_distances[landmark_distances != 0])

        # Logging edge success rates
        no_diagonal_edge_success_rate = edge_success_rate[~np.eye(self.num_landmarks, dtype=bool)]
        self.success_rates.append(np.mean(no_diagonal_edge_success_rate))
        self.non_zero_success_rates.append(np.mean(no_diagonal_edge_success_rate[no_diagonal_edge_success_rate > 0]))
        
        total_edges = max((np.sum(no_diagonal_edge_success_rate > edge_threshold)) // 2, 0)
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
                if self.use_affinity and success_rate == 0:
                    dist = -1 * np.log(zero_success_dist * 1e-2)
                
                landmark_distances[(u, v)] = dist
                # landmark_distances[(u, v)] = 1.001 - success_rate
                
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

    def get_true_edges(self, nodes, weight):
        w = 0
        for i, node in enumerate(nodes[1:]):
            prev = nodes[i]
            if self.graph[prev][node][weight] < (self.max_landmarks * 3):
                w += 1
        return w 

    def get_path_weight(self, nodes, weight):
        w = 0
        for i, node in enumerate(nodes[1:]):
            prev = nodes[i]
            w += self.graph[prev][node][weight]
        return w

    def generate_path(self, source, target, mode):
        # Generate path from source to target in landmark graph
        
        # Select path with probability given by softmin of path lengths
        if self.landmark_paths is not None:
            paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, source, target, weight='weight'), self.landmark_paths))
            if mode != 'eval':
                path_weights = np.arange(1, len(paths) + 1)
            else:
                path_weights = np.array([self.get_path_weight(path, 'weight') for path in paths])

            # if mode == 'eval':
            #     true_edges = np.array([self.get_true_edges(path, 'weight') for path in paths])
            #     max_true_edges = max(true_edges)
            #     paths = [path for i, path in enumerate(paths) if true_edges[i] == max_true_edges]

            path_p = softmax(-1 * path_weights)

            if mode == 'eval':
                self.eval_paths = paths
                self.eval_paths_p = path_p
            path_choice = np.random.choice(list(range(len(paths))), p=path_p)
            self.path = paths[path_choice]
        
        # Select shortest path that has most number of actual nodes
        else:
            max_length = 0
            for path in nx.all_shortest_paths(self.graph, source, target, weight='weight'):
                if len(path) > max_length:
                    self.path = path
                    max_length = len(path)
            self.eval_paths = [self.path]
            self.eval_paths_p = np.array([1])
        return self.path

    def update_affinity(self):
        if self.successes is not None:
            self.successes = (self.successes * self.affinity_decay)
        if self.attempts is not None:
            self.attempts = (self.attempts * self.affinity_decay)

    def prune_landmarks(self):
        # Prune landmarks that do not meet similarity requirement [NOT CURRENTLY USED]
        # landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
        # save_idx = torch.sum(landmark_similarities < self.threshold, axis=1) >= (self.num_landmarks - 1)

        seen_positions = set()
        save_idx = []
        for i, position, action in zip(range(self.num_landmarks), map(tuple, self.positions), self.actions):
            if (*position, action) not in seen_positions:
                seen_positions.add((*position, action))
                save_idx.append(i)
        save_idx = np.array(save_idx)

        self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]
        self.positions = self.positions[save_idx]
        self.actions = self.actions[save_idx]
        
        self.visitations = self.visitations[save_idx]
        self.successes = self.successes[save_idx[:, None], save_idx]
        self.attempts = self.attempts[save_idx[:, None], save_idx]

        self.landmark_removes += (self.num_landmarks - len(save_idx))
        self.num_landmarks = len(save_idx)
