import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import floyd_warshall
from sklearn_extra.cluster import KMedoids
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.idf_dsr_agent import IDFDSRAgent, AgentInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

Landmarks = namedarraytuple("Landmarks", ["observation", "features", "dsr"])

# class Landmarks(object):

#     def __init__(self, max_landmarks, tau=0.9):
#         save__init__args(locals())
#         self.num_landmarks = 0
#         self.observation = None 
#         self.feature = None
#         self.norm_feature = None
#         self.dsr = None
#         self.norm_dsr = None
#         self.visitation = None

#     def add_landmark(self, observation, feature, dsr):
#         if self.num_landmarks == 0:
#             self.observation = np.expand_dims(observation, 0)
#             self.feature = feature
#             self.norm_feature = feature / torch.norm(feature, ord=2, keepdim=True)
#             self.dsr = dsr
#             self.norm_dsr = dsr / torch.norm(dsr, ord=2, keepdim=True)
#         else:
#             normed_dsr = dsr / torch.norm(dsr, ord=2, keepdim=True)
#             similarity = torch.dot(normed_dsr, self.norm_dsr)  # cosine similarity of dsr

#             if any(similarity > self.tau):
#                 if self.num_landmarks < self.max_landmarks:
#                     norm_feature = feature / torch.norm(feature, ord=2, keepdim=True)
#                     self.observation = np.append(self.observation, )
#                     self.num_landmarks += 1


class LandmarkAgent(IDFDSRAgent):

    def __init__(
            self,
            exploit_prob=0.01,
            landmark_update_interval=int(5e3),
            landmark_batch_size=2048,
            landmark_distance_threshold=1,
            num_landmarks=8,
            steps_per_landmark=25,
            **kwargs):
        save__init__args(locals())
        self.explore = True
        # self.skm = SphericalKMeans(n_clusters=num_landmarks)
        self.kmedoids = KMedoids(n_clusters=num_landmarks)  # TODO: what metric to use?
        self.landmarks = None
        super().__init__(**kwargs)

    @torch.no_grad()
    def update_landmarks(self, itr):
        if (itr + 1) % self.landmark_update_interval == 0:
            observation = self._replay_buffer.sample_batch(self.landmark_batch_size).agent_inputs.observation

            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            reshaped_dsr = dsr.detach().cpu().numpy().reshape(dsr.shape[0], -1)

            self.kmedoids.fit(reshaped_dsr)
            landmark_indices = self.kmedoids.medoid_indices_
            
            # Remove empty clusters
            for i in range(len(landmark_indices)):
                if self.kmedoids.labels_[landmark_indices[i]] != i:
                    landmark_indices[i] = -1
            landmark_indices = landmark_indices[landmark_indices != -1]
            self.num_landmarks = len(landmark_indices)

            landmark_features = features[landmark_indices] 
            landmark_features /= torch.norm(landmark_features, p=2, keepdim=True, dim=1)

            landmark_dsr = dsr[landmark_indices].mean(dim=1)

            self.landmarks = Landmarks(
                observation=observation[landmark_indices],
                features=landmark_features,
                dsr=landmark_dsr
            )

            # TODO:  perhaps initialize visitation counts based on old landmarks via nearest neighbors?
            self.landmark_visitations = np.zeros(self.num_landmarks)  

            landmark_dsr = landmark_dsr.detach().cpu().numpy()

            landmark_distances = distance_matrix(landmark_dsr, landmark_dsr)
            _, predecessors = floyd_warshall(landmark_distances, directed=False, return_predecessors=True)
            self.landmark_predecessor = predecessors

    def set_replay_buffer(self, replay_buffer):
        self._replay_buffer = replay_buffer

    @property
    def replay_buffer(self):
        return self._replay_buffer

    def exploit(self):
        if np.random.random() < self.exploit_prob and self._mode != 'eval' and self.explore and self.landmarks:
            self.explore = False
            self.landmark_steps = 0

            inverse_visitations = 1. / (self.landmark_visitations + 1e-6)
            landmark_probabilities = inverse_visitations / inverse_visitations.sum()
            self.goal_landmark = np.random.choice(range(self.num_landmarks), p=landmark_probabilities)

            self.subgoal_landmark = None

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        self.exploit()

        if not self.explore:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            if self.subgoal_landmark is None:
                landmark_distances = torch.norm(self.landmarks.dsr - dsr.mean(dim=1), p=2, dim=1)
                self.subgoal_landmark = landmark_distances.argmin().item()
                self.landmark_visitations[self.subgoal_landmark] += 1

            if self.landmark_steps < self.steps_per_landmark:
                subgoal_distance = torch.norm(self.landmarks.dsr[self.subgoal_landmark] - dsr.mean(dim=1), p=2, dim=1)
                if subgoal_distance < self.landmark_distance_threshold:
                    if self.subgoal_landmark == self.goal_landmark:
                        self.explore = True
                    else:
                        next_landmark = self.landmark_predecessor[self.subgoal_landmark, self.goal_landmark]
                        if self.subgoal_landmark == next_landmark:
                            self.subgoal_landmark = self.goal_landmark
                        else:
                            self.subgoal_landmark = next_landmark
                        self.landmark_steps = 0
                        self.landmark_visitations[self.subgoal_landmark] += 1

            else:
                if self.subgoal_landmark == self.goal_landmark:
                    self.explore = True
                else:
                    next_landmark = self.landmark_predecessor[self.subgoal_landmark, self.goal_landmark]
                    if self.subgoal_landmark == next_landmark or next_landmark == -9999:
                        self.subgoal_landmark = self.goal_landmark
                    else:
                        self.subgoal_landmark = next_landmark
                    self.landmark_steps = 0
                    self.landmark_visitations[self.subgoal_landmark] += 1

        if self.explore:
            action = torch.randint_like(prev_action, high=self.distribution.dim)
        else:
            subgoal_landmark_features = self.landmarks.features[self.subgoal_landmark]
            q_values = torch.matmul(dsr, subgoal_landmark_features).cpu()
            action = self.distribution.sample(q_values)
            self.landmark_steps += 1

        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)
