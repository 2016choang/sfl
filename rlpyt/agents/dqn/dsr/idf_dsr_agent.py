
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

    def __init__(self, idf_model_kwargs=None, initial_idf_model_state_dict=None, **kwargs):
        save__init__args(locals())
        ModelCls = GridDsrModel
        super().__init__(ModelCls=ModelCls, **kwargs)

        self.IDFModelCls = IDFModel
        if self.idf_model_kwargs is None:
            self.idf_model_kwargs = dict()

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.idf_model.to(self.device)
        self.idf_model.to(self.device)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.idf_model = self.IDFModelCls(**self.env_model_kwargs,
            **self.idf_model_kwargs)
        if self.initial_idf_model_state_dict is not None:
            self.idf_model.load_state_dict(self.initial_idf_model_state_dict)

        self.target_idf_model = self.IDFModelCls(**self.env_model_kwargs,
            **self.idf_model_kwargs)
        self.target_idf_model.load_state_dict(self.idf_model.state_dict())

    def encode(self, observation):
        model_inputs = buffer_to(observation,
            device=self.device)
        features = self.idf_model(model_inputs, mode='encode')
        return features.cpu()

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict(),
            idf_model=self.idf_model.state_dict(),
            target_idf=self.target_idf_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
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
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def inverse_dynamics(self, observation, next_observation):
        model_inputs = buffer_to((observation, next_observation),
            device=self.device)
        pred_actions = self.idf_model(*model_inputs, mode='inverse')
        return pred_actions.cpu()

    def update_idf(self):
        self.target_idf_model.load_state_dict(
            strip_ddp_state_dict(self.idf_model.state_dict()))

    def dsr_parameters(self):
        return [param for name, param in self.model.named_parameters()]

    def idf_parameters(self):
        return [param for name, param in self.idf_model.named_parameters()]
