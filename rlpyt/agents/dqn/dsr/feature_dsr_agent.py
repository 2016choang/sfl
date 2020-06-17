
import numpy as np
from sklearn.manifold import TSNE
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dsr.dsr_agent import DsrAgent, AgentInfo
from rlpyt.agents.dqn.mixin import Mixin
from rlpyt.models.dqn.dsr.idf_model import IDFModel
from rlpyt.models.dqn.dsr.tcf_model import TCFModel
from rlpyt.models.dqn.dsr.grid_dsr_model import GridDsrModel
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.quick_args import save__init__args


class FeatureDSRAgent(Mixin, DsrAgent):

    def __init__(self, feature_model_kwargs={}, initial_feature_model_state_dict=None, **kwargs):
        save__init__args(locals())
        ModelCls = GridDsrModel
        super().__init__(ModelCls=ModelCls, **kwargs)

        self.featureModelCls = None

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.feature_model.to(self.device)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.feature_model = self.featureModelCls(**self.env_model_kwargs,
            **self.feature_model_kwargs)
        if self.initial_feature_model_state_dict is not None:
            self.feature_model.load_state_dict(self.initial_feature_model_state_dict)

    def encode(self, observation):
        # Encode observation into feature representation
        model_inputs = buffer_to(observation,
            device=self.device)
        features = self.feature_model(model_inputs, mode='encode')
        return features.cpu()

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict(),
            feature_model=self.feature_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        if self.distribution.epsilon >= 1.0:
            # Random policy
            action = torch.randint_like(prev_action, high=self.distribution.dim)
        else:
            # Epsilon-greedy over q-values generated with SF
            model_inputs = buffer_to(observation,
                device=self.device)
            features = self.feature_model(model_inputs, mode='encode')

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

    def dsr_parameters(self):
        return [param for name, param in self.model.named_parameters()]

    def feature_parameters(self):
        return [param for name, param in self.feature_model.named_parameters()]

    @torch.no_grad()
    def get_representations(self, env):
        # Get features and SFs of possible observations
        h, w = env.grid.height, env.grid.width
        features = torch.zeros((h, w, 4, self.feature_model.feature_size), dtype=torch.float)
        features += np.nan
        dsr = torch.zeros((h, w, 4, env.action_space.n, self.model.feature_size), dtype=torch.float)
        dsr += np.nan

        for pos in env.get_possible_pos():
            x, y = pos
            for direction in range(4):
                env.unwrapped.agent_pos = np.array([x, y])
                env.unwrapped.agent_dir = direction
                obs, _, _, _ = env.get_current_state()

                model_inputs = buffer_to(torch.Tensor(obs).unsqueeze(0),
                    device=self.device)

                features[x, y, direction] = self.feature_model(model_inputs, mode='encode')

                model_inputs = buffer_to(features[x, y, direction],
                    device=self.device)

                dsr[x, y, direction] = self.model(model_inputs, mode='dsr')

        return features, dsr

    @torch.no_grad()
    def get_representation_heatmap(self, representation, subgoal=(4, 13), mean_axes=(2, 3), distance='cos'):
        representation = representation.detach().numpy()
        representation_matrix = representation.mean(axis=mean_axes)
        representation_matrix = representation_matrix / np.linalg.norm(representation_matrix, ord=2, axis=2, keepdims=True)

        subgoal_representation = representation_matrix[subgoal]
        side_size = representation_matrix.shape[0]
        heatmap = np.zeros((side_size, side_size))
        for x in range(side_size):
            for y in range(side_size):
                if distance == 'cos':
                    heatmap[x, y] = np.dot(representation_matrix[x, y], subgoal_representation)
                elif distance == 'l2':
                    heatmap[x, y] = np.linalg.norm(representation_matrix[x, y] - subgoal_representation, ord=2)
                else:
                    raise NotImplementedError
        return heatmap

    @torch.no_grad()
    def get_q_values(self, env, dsr, subgoal=(4, 13), mean_axes=(2, )):
        dsr = dsr.detach().numpy()
        dsr_matrix = dsr.mean(axis=mean_axes)
        dsr_matrix = dsr_matrix / np.linalg.norm(dsr_matrix, ord=2, axis=3, keepdims=True)
        subgoal_dsr = dsr_matrix[subgoal].mean(axis=0)
        q_values = np.dot(dsr_matrix, subgoal_dsr)
        return q_values

    @torch.no_grad()
    def get_tsne(self, env, representation, mean_axes=(2, 3)):
        h, w = env.grid.height, env.grid.width
        representation = representation.detach().numpy()
        representation_matrix = np.nanmean(representation, axis=mean_axes)

        valid_representations = representation_matrix.reshape(h * w, -1)
        walls = np.isnan(valid_representations).any(axis=1)
        valid_representations = valid_representations[~walls]

        embeddings = TSNE(n_components=2).fit_transform(valid_representations)

        rooms = np.zeros((h, w))
        if hasattr(env, 'rooms'):
            for i, room in enumerate(env.rooms, 1):
                start_x, start_y = room.top
                size_x, size_y = room.size
                for x in range(start_x + 1, start_x + size_x - 1):
                    for y in range(start_y + 1, start_y + size_y - 1):
                        rooms[x, y] = i
        rooms = rooms.reshape(h * w)[~walls]
        return embeddings, rooms

    def train_mode(self, itr):
        super().train_mode(itr)
        self.feature_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.feature_model.eval()

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.feature_model.eval()
    

class IDFDSRAgent:

    def __init__(self):
        self.featureModelCls = IDFModel

    def inverse_dynamics(self, observation, next_observation):
        model_inputs = buffer_to(observation,
            device=self.device)

        features = self.feature_model(model_inputs, mode='encode')

        model_inputs = buffer_to(next_observation,
            device=self.device)
        next_features = self.feature_model(model_inputs, mode='encode')

        model_inputs = buffer_to((features, next_features),
            device=self.device)
        pred_actions = self.feature_model(*model_inputs, mode='inverse')
        return pred_actions.cpu()


class TCFDSRAgent:

    def __init__(self):
        self.featureModelCls = TCFModel