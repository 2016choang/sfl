
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.dqn.grid_dsr_model import GridDsrModel, GridDsrDummyModel
from rlpyt.models.utils import strip_ddp_state_dict


AgentInfo = namedarraytuple("AgentInfo", "a")


class TabularDsrAgent(EpsilonGreedyAgentMixin, BaseAgent):

    def __init__(self, initial_M=None, **kwargs):
        self.initial_M = initial_M
        super().__init__(ModelCls=GridDsrDummyModel, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

    def __call__(self, observation):
        model_inputs = buffer_to(observation,
            device=self.device)
        return self.M[:, model_inputs.argmax(dim=1), :].permute(1, 0, 2).cpu()

    def encode(self, observation):
        observation = buffer_to(observation,
            device=self.device)
        return observation.type(torch.float).cpu()

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.state_size = env_spaces.observation.shape[0]
        self.action_size = env_spaces.action.n
        if self.initial_M is not None:
            self.M = self.initial_M
        else:
            self.M = torch.stack([torch.eye(self.state_size) for _ in range(self.action_size)])
        self.target_model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.distribution = EpsilonGreedy(dim=self.action_size)
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.M = self.M.to(self.device)
        self.target_model.to(self.device)

    def state_dict(self):
        return self.M

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        observation = observation.reshape(-1, self.state_size)
        action = torch.randint(high=self.action_size, size=(observation.shape[0],))
        agent_info = AgentInfo(a=action)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_M(self, action, state, td, learning_rate):
        action = buffer_to(action, 
            device=self.device)
        state = buffer_to(state,
            device=self.device)
        td = buffer_to(td,
            device=self.device)
        self.M[action, state] += (learning_rate * td)

    def target(self, observation):
        model_inputs = buffer_to(observation,
            device=self.device)
        return self.M[:, model_inputs.argmax(dim=1), :].permute(1, 0, 2).cpu()

    def update_target(self):
        return


class TabularFeaturesDsrAgent(EpsilonGreedyAgentMixin, BaseAgent):

    def __init__(self, initial_M=None, **kwargs):
        self.initial_M = initial_M
        super().__init__(ModelCls=GridDsrDummyModel, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

    def __call__(self, observation):
        model_inputs = buffer_to(observation.position,
            device=self.device)
        return self.M[:, model_inputs.argmax(dim=1), :].permute(1, 0, 2).cpu()

    def encode(self, observation):
        observation = buffer_to(observation,
            device=self.device)
        return observation._replace(features=observation.features.type(torch.float).cpu(),
            position=observation.position.type(torch.float).cpu())

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.state_size = env_spaces.observation.shape.position[0]
        self.feature_size = env_spaces.observation.shape.features[0]
        self.action_size = env_spaces.action.n
        if self.initial_M is not None:
            self.M = self.initial_M
        else:
            self.M = torch.stack([torch.zeros((self.state_size, self.feature_size)) for _ in range(self.action_size)])
        self.target_model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.distribution = EpsilonGreedy(dim=self.action_size)
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.M = self.M.to(self.device)
        self.target_model.to(self.device)

    def state_dict(self):
        return self.M

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        observation = observation.position.reshape(-1, self.state_size)
        action = torch.randint(high=self.action_size, size=(observation.shape[0],))
        agent_info = AgentInfo(a=action)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_M(self, action, state, td, learning_rate):
        action = buffer_to(action, 
            device=self.device)
        state = buffer_to(state,
            device=self.device)
        td = buffer_to(td,
            device=self.device)
        self.M[action, state] += (learning_rate * td)

    def target(self, observation):
        model_inputs = buffer_to(observation.position,
            device=self.device)
        return self.M[:, model_inputs.argmax(dim=1), :].permute(1, 0, 2).cpu()

    def update_target(self):
        return
