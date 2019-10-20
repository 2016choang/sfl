
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.utils import strip_ddp_state_dict


AgentInfo = namedarraytuple("AgentInfo", "a")


class DsrAgent(EpsilonGreedyAgentMixin, BaseAgent):

    def __call__(self, features):
        model_inputs = buffer_to(features, device=self.device)
        dsr = self.model(model_inputs, mode='dsr')
        return dsr.cpu()

    def encode(self, observation):
        model_inputs = buffer_to(observation, device=self.device)
        features = self.model(model_inputs)
        return features.cpu()
    
    def reconstruct(self, observation):
        model_inputs = buffer_to(observation, device=self.device)
        reconstructed = self.model(model_inputs, mode='reconstruct')
        return reconstructed.cpu()

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.target_model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.num_actions = env_spaces.action.n
        self.distribution = EpsilonGreedy(dim=env_spaces.action.n)  # unused
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.target_model.to(self.device)

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        action = torch.randint_like(prev_action, high=self.num_actions)
        agent_info = AgentInfo(a=action)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, features):
        model_inputs = buffer_to(features, device=self.device)
        target_dsr = self.model(model_inputs, mode='dsr')
        return target_dsr.cpu()

    def update_target(self):
        self.target_model.load_state_dict(
            strip_ddp_state_dict(self.model.state_dict()))
