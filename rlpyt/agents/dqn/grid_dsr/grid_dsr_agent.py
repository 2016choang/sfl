
from rlpyt.agents.dqn.dsr_agent import DsrAgent
from rlpyt.models.dqn.grid_dsr_model import GridDsrModel, GridDsrSmallModel, GridDsrCompactModel, GridDsrRandomModel
from rlpyt.agents.dqn.grid_dsr.mixin import GridMixin


class GridDsrAgent(GridMixin, DsrAgent):

    def __init__(self, ModelCls=GridDsrModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class GridDsrSmallAgent(GridMixin, DsrAgent):

    def __init__(self, ModelCls=GridDsrSmallModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class GridDsrCompactAgent(GridMixin, DsrAgent):

    def __init__(self, ModelCls=GridDsrCompactModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

class GridDsrRandomAgent(GridMixin, DsrAgent):

    def __init__(self, ModelCls=GridDsrRandomModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
