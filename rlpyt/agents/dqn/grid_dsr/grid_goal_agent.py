
from rlpyt.agents.dqn.dsr_agent import DsrAgent
from rlpyt.models.dqn.grid_dsr_model import GridGoalModel
from rlpyt.agents.dqn.grid_dsr.mixin import GridMixin
from rlpyt.utils.buffer import buffer_to


class GridGoalAgent(GridMixin, DsrAgent):

    def __init__(self, **kwargs):
        ModelCls = GridGoalModel
        super().__init__(ModelCls=ModelCls, **kwargs)

    def embed_goal(self, observation):
        model_inputs = buffer_to(observation,
            device=self.device)
        goal_embedding = self.model(model_inputs, mode='goal')
        return goal_embedding.cpu()

    def goal_parameters(self):
        return [param for name, param in self.model.named_parameters() if 'goal' in name]
