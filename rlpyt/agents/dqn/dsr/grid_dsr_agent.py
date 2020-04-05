
from rlpyt.agents.dqn.dsr.dsr_agent import DsrAgent
from rlpyt.models.dqn.dsr.grid_dsr_model import GridDsrModel, GridActionDsrModel
from rlpyt.agents.dqn.mixin import Mixin


class GridDsrAgent(Mixin, DsrAgent):

    def __init__(self, mode=None, **kwargs):
        if 'action' in mode:
            ModelCls = GridActionDsrModel
        else:
            ModelCls = GridDsrModel
        super().__init__(ModelCls=ModelCls, **kwargs)
