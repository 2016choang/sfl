
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
        self.mean = self.mean.to(self.device)
        self.var = self.var.to(self.device)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.idf_model = self.IDFModelCls(**self.env_model_kwargs,
            **self.idf_model_kwargs)
        if self.initial_idf_model_state_dict is not None:
            self.idf_model.load_state_dict(self.initial_idf_model_state_dict)
        self.mean = torch.zeros(self.idf_model_kwargs['feature_size'])
        self.var = torch.ones(self.idf_model_kwargs['feature_size'])

    def encode(self, observation, normalize=True):
        model_inputs = buffer_to(observation,
            device=self.device)
        features = self.idf_model(model_inputs, mode='encode')
        if normalize:
            with torch.no_grad():
                features = (features - self.mean) / (self.var + self.epsilon).sqrt()                
        return features.cpu()

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict(),
            idf_model=self.idf_model.state_dict(),
            mean=self.mean,
            var=self.var)

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
        with torch.no_grad():
            N = features.shape[0]
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * features.mean(axis=0)
            self.var = self.momentum  * self.var + (1.0 - self.momentum) * (N / (N - 1) * features.var(axis=0))

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
