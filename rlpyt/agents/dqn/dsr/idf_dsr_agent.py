
import numpy as np
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.dsr_agent import DsrAgent, AgentInfo
from rlpyt.agents.dqn.mixin import Mixin
from rlpyt.models.dqn.dsr.idf_model import IDFModel
from rlpyt.models.dqn.dsr.grid_dsr_model import GridDsrModel
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.quick_args import save__init__args


class IDFDSRAgent(Mixin, DsrAgent):

    def __init__(self, idf_model_kwargs=None, initial_idf_model_state_dict=None, momentum=0.9, epsilon=1e-5, **kwargs):
        save__init__args(locals())
        ModelCls = GridDsrModel
        super().__init__(ModelCls=ModelCls, **kwargs)

        self.IDFModelCls = IDFModel
        if self.idf_model_kwargs is None:
            self.idf_model_kwargs = dict()

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.idf_model.to(self.device)
        # self.mean = self.mean.to(self.device)
        # self.var = self.var.to(self.device)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.idf_model = self.IDFModelCls(**self.env_model_kwargs,
            **self.idf_model_kwargs)
        if self.initial_idf_model_state_dict is not None:
            self.idf_model.load_state_dict(self.initial_idf_model_state_dict)
        # self.mean = torch.zeros(self.idf_model_kwargs['feature_size'])
        # self.var = torch.ones(self.idf_model_kwargs['feature_size'])

    def encode(self, observation, normalize=True):
        model_inputs = buffer_to(observation,
            device=self.device)
        features = self.idf_model(model_inputs, mode='encode')
        # if normalize:
        #     with torch.no_grad():
        #         features = (features - self.mean) / (self.var + self.epsilon).sqrt()                
        return features.cpu()

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict(),
            idf_model=self.idf_model.state_dict())


    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        # random exploration policy shortcut
        if self.distribution.epsilon >= 1.0:
            if prev_action.shape:
                q = torch.zeros(prev_action.shape[0], self.distribution.dim)
            else:
                q = torch.zeros(self.distribution.dim)
        else:
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.idf_model(model_inputs, mode='encode')

            model_inputs = buffer_to(features,
                device=self.device)
            dsr = self.model(model_inputs, mode='dsr')

            model_inputs = buffer_to(dsr,
                device=self.device)
            q = self.model(model_inputs, mode='q')
            q = q.cpu()

        action = self.distribution.sample(q)
        agent_info = AgentInfo(a=action)
        return AgentStep(action=action, agent_info=agent_info)

    def inverse_dynamics(self, observation, next_observation):
        model_inputs = buffer_to(observation,
            device=self.device)

        features = self.idf_model(model_inputs, mode='encode')
        # with torch.no_grad():
        #     N = features.shape[0]
        #     self.mean = self.momentum * self.mean + (1.0 - self.momentum) * features.mean(axis=0)
        #     self.var = self.momentum  * self.var + (1.0 - self.momentum) * (N / (N - 1) * features.var(axis=0))

        model_inputs = buffer_to(next_observation,
            device=self.device)
        next_features = self.idf_model(model_inputs, mode='encode')

        model_inputs = buffer_to((features, next_features),
            device=self.device)
        pred_actions = self.idf_model(*model_inputs, mode='inverse')
        return pred_actions.cpu()

    def dsr_parameters(self):
        return [param for name, param in self.model.named_parameters()]

    def idf_parameters(self):
        return [param for name, param in self.idf_model.named_parameters()]

    @torch.no_grad()
    def get_dsr(self, env):
        h, w = env.grid.height, env.grid.width
        dsr = torch.zeros((h, w, 4, env.action_space.n, self.idf_model.feature_size), dtype=torch.float)
        dsr += np.nan

        for room in env.rooms:
            start_x, start_y = room.top
            size_x, size_y = room.size
            for direction in range(4):
                for x in range(start_x + 1, start_x + size_x - 1):
                    for y in range(start_y + 1, start_y + size_y - 1):
                        env.env.env.unwrapped.agent_pos = np.array([x, y])
                        env.env.env.unwrapped.agent_dir = direction
                        obs, _, _, _ = env.env.env.step(5)
                        
                        model_inputs = buffer_to(torch.Tensor(obs).unsqueeze(0),
                            device=self.device)

                        features = self.idf_model(model_inputs, mode='encode')

                        model_inputs = buffer_to(features,
                            device=self.device)

                        dsr[x, y, direction] = self.model(model_inputs, mode='dsr')

                if room.exitDoorPos is not None:
                    exit_door = np.array(room.exitDoorPos)
                    env.env.env.unwrapped.agent_pos = exit_door
                    env.env.env.unwrapped.agent_dir = direction
                    obs, _, _, _ = env.env.env.step(5)

                    model_inputs = buffer_to(torch.Tensor(obs).unsqueeze(0),
                            device=self.device)

                    features = self.idf_model(model_inputs, mode='encode')

                    model_inputs = buffer_to(features,
                        device=self.device)

                    dsr[exit_door[0], exit_door[1], direction] = self.model(model_inputs, mode='dsr')

        return dsr

    @torch.no_grad()
    def get_dsr_heatmap(self, dsr, subgoal=(4, 13), direction=-1, action=-1):
        dsr = dsr.detach().numpy()

        if direction == -1:
            dsr_matrix = dsr.mean(axis=2)
            
        else:
            dsr_matrix = dsr[:, :, direction]

        if action == -1:
            dsr_matrix = dsr_matrix.mean(axis=2)
        else:
            dsr_matrix = dsr_matrix[:, :, action]
        
        dsr_matrix = dsr_matrix / np.linalg.norm(dsr_matrix, ord=2, axis=2, keepdims=True)

        subgoal_dsr = dsr_matrix[subgoal]

        side_size = dsr_matrix.shape[0]
        heatmap = np.zeros((side_size, side_size))
        for x in range(side_size):
            for y in range(side_size):
                heatmap[x, y] = np.dot(dsr_matrix[x, y], subgoal_dsr)

        return heatmap

    @torch.no_grad()
    def get_q_values(self, env, dsr, subgoal=(4, 13), direction=-1):
        dsr = dsr.detach().numpy()

        if direction == -1:
            dsr_matrix = dsr.mean(axis=2)
            
        else:
            dsr_matrix = dsr[:, :, direction]
        
        dsr_matrix = dsr_matrix / np.linalg.norm(dsr_matrix, ord=2, axis=3, keepdims=True)

        side_size = dsr_matrix.shape[0]

        env.unwrapped.agent_pos = np.array(subgoal)
        obs = env.env.env.step(5)[0]
        obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        features = self.idf_model(obs, mode='encode')
        features = features.squeeze().detach().cpu().numpy()
        normed_features = features / np.linalg.norm(features, ord=2)

        q_values = np.dot(dsr_matrix, normed_features)

        return q_values
