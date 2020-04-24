import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import floyd_warshall
from sklearn_extra.cluster import KMedoids
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.idf_dsr_agent import IDFDSRAgent, AgentInfo
from rlpyt.utils.buffer import buffer_to, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

class Landmarks(object):

    def __init__(self, max_landmarks, threshold=0.75):
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

    def add_goal_landmark(self, observation, features, dsr):
        self.observations = torch.cat((self.observations, observation), dim=0)
        self.set_features(features, self.num_landmarks)
        self.set_dsr(dsr, self.num_landmarks)
        self.num_landmarks += 1
        self.visitations = np.append(self.visitations, 0)

    def remove_goal_landmark(self):
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
        else:
            norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True)
            similarity = torch.matmul(self.norm_dsr, norm_dsr.T)  # cosine similarity of dsr

            if sum(similarity < self.threshold) >= max((self.num_landmarks // 2), 1):
                observation = observation
                if self.num_landmarks < self.max_landmarks:
                    self.observations = torch.cat((self.observations, observation), dim=0)
                    self.set_features(features, self.num_landmarks)
                    self.set_dsr(dsr, self.num_landmarks)
                    self.num_landmarks += 1
                    self.visitations = np.append(self.visitations, 0)

                else:
                    landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
                    replace_idx = torch.sum(landmark_similarities < self.threshold, axis=1).argmin().item()
                    self.observations[replace_idx] = observation
                    self.set_features(features, replace_idx)
                    self.set_dsr(dsr, replace_idx)
                    self.visitations[replace_idx] = 0

    def set_features(self, features, idx=None):
        features = features
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
        self.path = nx.shortest_path(self.graph, source, target, weight='weight')
        return self.path

    def prune_landmarks(self):
        landmark_similarities = torch.matmul(self.norm_dsr, self.norm_dsr.T)
        save_idx = torch.sum(landmark_similarities < self.threshold, axis=1) >= (self.num_landmarks // 2)

        self.observations = self.observations[save_idx]
        self.features = self.features[save_idx]
        self.norm_features = self.norm_features[save_idx]
        self.dsr = self.dsr[save_idx]
        self.norm_dsr = self.norm_dsr[save_idx]
        
        save_idx = save_idx.detach().cpu().numpy()
        self.visitations = self.visitations[save_idx]

        self.num_landmarks = sum(save_idx)


class LandmarkAgent(IDFDSRAgent):

    def __init__(
            self,
            exploit_prob=0.01,
            landmark_update_interval=int(5e3),
            add_threshold=0.75,
            reach_threshold=0.95,
            max_landmarks=8,
            steps_per_landmark=25,
            **kwargs):
        save__init__args(locals())
        self.explore = True
        self.landmarks = None
        self.update = False
        self.eval_goal = False
        super().__init__(**kwargs)

    def update_landmarks(self, itr):
        if (itr + 1) % self.landmark_update_interval == 0:
            if self.landmarks is None:
                self.landmarks = Landmarks(self.max_landmarks, self.add_threshold)
            elif self.landmarks.num_landmarks:
                self.update = True

    def exploit(self):
        if self.landmarks is not None and self.landmarks.num_landmarks and self._mode != 'eval' and \
                self.explore: 
            if self.update:
                self.update = False
                observation = self.landmarks.observations

                model_inputs = buffer_to(observation,
                    device=self.device)
                features = self.idf_model(model_inputs, mode='encode')
                self.landmarks.set_features(features)

                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')
                self.landmarks.set_dsr(dsr)

                self.landmarks.prune_landmarks()
 
            if np.random.random() < self.exploit_prob:
                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None

                inverse_visitations = 1. / (self.landmarks.visitations + 1e-6)
                landmark_probabilities = inverse_visitations / inverse_visitations.sum()
                self.goal_landmark = np.random.choice(range(len(landmark_probabilities)), p=landmark_probabilities)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        self.exploit()

        if self.landmarks is not None:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            # only add landmarks in explore phase
            if self.explore and self._mode != 'eval':
                self.landmarks.add_landmark(observation, features, dsr)

            if not self.explore:
                if self.current_landmark is None:
                    norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                    landmark_similarity = torch.matmul(self.landmarks.norm_dsr, norm_dsr.T)
                    self.current_landmark = landmark_similarity.argmax().item()
                    self.landmarks.generate_graph()
                    self.path = self.landmarks.generate_path(self.current_landmark, self.goal_landmark)
                    self.path_idx = 0

                if self.landmark_steps < self.steps_per_landmark:
                    norm_dsr = dsr.mean(dim=1) / torch.norm(dsr.mean(dim=1), p=2, keepdim=True) 
                    subgoal_similarity = torch.matmul(self.landmarks.norm_dsr[self.current_landmark], norm_dsr.T)
                    if subgoal_similarity > self.reach_threshold:
                        self.landmarks.visitations[self.current_landmark] += 1
                        if self.current_landmark == self.goal_landmark:
                            self.explore = True
                        else:
                            self.path_idx += 1
                            self.current_landmark = self.path[self.path_idx]
                            self.landmark_steps = 0

                else:
                    self.landmarks.visitations[self.current_landmark] += 1
                    if self.current_landmark == self.goal_landmark:
                        self.explore = True
                    else:
                        self.path_idx += 1
                        try:
                            self.current_landmark = self.path[self.path_idx]
                        except IndexError:
                            import pdb; pdb.set_trace()
                        self.landmark_steps = 0

        if self.explore:
            action = torch.randint_like(prev_action, high=self.distribution.dim)
        else:
            subgoal_landmark_features = self.landmarks.norm_features[self.current_landmark]
            q_values = torch.matmul(dsr, subgoal_landmark_features).cpu()
            action = self.distribution.sample(q_values)
            self.landmark_steps += 1

        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def set_eval_goal(self, goal_obs):
        if self.landmarks is not None:
            observation = torchify_buffer(goal_obs).unsqueeze(0).float()

            model_inputs = buffer_to(observation,
                    device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                    device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            self.landmarks.add_goal_landmark(observation, features, dsr)        

            self.landmark_state = {'explore': self.explore,
                                   'landmark_steps': self.landmark_steps,
                                   'subgoal_landmark': self.current_landmark,
                                   'goal_landmark': self.goal_landmark}
            
            self.eval_goal = True

    def reset(self):
        if self.landmarks is not None:
            if self._mode == 'eval':
                self.explore = False
                self.landmark_steps = 0
                self.current_landmark = None
                self.goal_landmark = self.landmarks.num_landmarks - 1
            else:
                self.explore = True

    def reset_one(self, idx):
        self.reset()

    @torch.no_grad()
    def remove_eval_goal(self):
        if self.landmarks is not None and self.eval_goal:
            self.landmarks.remove_goal_landmark()

            self.explore = self.landmark_state['explore']
            self.landmark_steps = self.landmark_state['landmark_steps']
            self.current_landmark = self.landmark_state['subgoal_landmark']
            self.goal_landmark = self.landmark_state['goal_landmark']

            self.eval_goal = False