
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel
from rlpyt.agents.dqn.mixin import Mixin


class AtariDqnAgent(Mixin, DqnAgent):

    def __init__(self, ModelCls=AtariDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
