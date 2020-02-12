
from rlpyt.agents.dqn.dsr_agent import DsrAgent
from rlpyt.models.dqn.grid_dsr_model import GridDsrModel, GridDsrSmallModel, GridDsrCompactModel, GridDsrRandomModel
from rlpyt.agents.dqn.grid_dsr.mixin import GridMixin


class GridDsrAgent(GridMixin, DsrAgent):

    def __init__(self, mode='full', **kwargs):
        if mode == 'full':
            ModelCls = GridDsrModel
        elif mode == 'small':
            ModelCls = GridDsrSmallModel
        elif mode == 'compact':
            ModelCls = GridDsrCompactModel
        elif mode == 'rooms' or mode == 'gaussian':
            ModelCls = GridDsrRandomModel
        super().__init__(ModelCls=ModelCls, **kwargs)
