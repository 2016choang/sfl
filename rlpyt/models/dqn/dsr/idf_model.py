import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import FUNCTION_MAP, Reshape


class IDFModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=64
            ):
        super().__init__()
        h, w, c = image_shape

        self.output_size = output_size
        self.feature_size = feature_size

        conv_embedding_size = 16 * (((h - 3) // 2) - 1) ** 2

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, (3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_embedding_size, self.feature_size)
        )

        self.inverse = nn.Sequential(
            nn.Linear(feature_size * 2, self.output_size)
        )

    def forward(self, obs, next_obs=None, mode='inverse'):
        x = obs.type(torch.float)
        if mode == 'inverse':
            next_x = next_obs.type(torch.float)
            return self.inverse(torch.cat((x, next_x), dim=1))
        elif mode == 'encode':
            x = x.permute(0, 3, 1, 2)
            return self.encoder(x)
        else:
            raise ValueError('Invalid mode!')
