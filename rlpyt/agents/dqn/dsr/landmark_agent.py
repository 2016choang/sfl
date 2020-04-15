import numpy as np
from sklearn_extra.cluster import KMedoids
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.idf_dsr_agent import IDFDSRAgent, AgentInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.quick_args import save__init__args


class LandmarkAgent(IDFDSRAgent):

    def __init__(self, seed, exploit_prob=0.01, exploit_batch_size=512, exploit_max_steps=25, num_landmarks=8, **kwargs):
        save__init__args(locals())
        self.explore = True
        self.kmedoids = KMedoids(n_clusters=num_landmarks, random_state=seed)
        super().__init__(**kwargs)

    def set_replay_buffer(self, replay_buffer):
        self._replay_buffer = replay_buffer

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @torch.no_grad()
    def exploit(self):
        if np.random.random() < self.exploit_prob and self._mode != 'eval' and self.explore:
            self.explore = False

            observation = self._replay_buffer.sample_batch(self.exploit_batch_size).agent_inputs.observation

            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            dsr = dsr.detach().cpu().numpy().reshape(dsr.shape[0], -1)

            import pdb; pdb.set_trace()
            # todo:
            # 1. start exploiting only after learning dsr
            # 2. avoid empty clusters
            # 3. test!
            self.kmedoids.fit(dsr)
            landmarks = self.kmedoids.medoid_indices_
            landmark_choice = np.random.randint(len(landmarks))

            self.landmark_features = features[landmark_choice]
            self.landmark_dsr = dsr[landmark_choice]
            self.exploit_steps = 0

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        self.exploit()

        if self.explore:
            action = torch.randint_like(prev_action, high=self.distribution.dim)
        else:
            # check if we are close enough to the landmark
            # landmark_similarity = torch.dot(dsr, self.landmark_dsr)
            if self.exploit_steps < self.exploit_max_steps:
                model_inputs = buffer_to(observation,
                    device=self.device)
                features = self.idf_model(model_inputs, mode='encode')

                model_inputs = buffer_to(features,
                    device=self.device)
                dsr = self.model(model_inputs, mode='dsr')

                q = torch.dot(dsr, self.landmark_features)

                action = self.distribution.sample(q)
                self.exploit_steps += 1

            else:   
                action = torch.randint_like(high=self.distribution.dim)
                self.explore = True
                
        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)
