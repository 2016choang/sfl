
from rlpyt.agents.dqn.dsr_agent import DsrAgent
from rlpyt.models.dqn.grid_dsr_model import GridDsrModel
from rlpyt.agents.dqn.grid_dsr.mixin import GridMixin


class GridDsrAgent(GridMixin, DsrAgent):

    def __init__(self, ModelCls=GridDsrModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
