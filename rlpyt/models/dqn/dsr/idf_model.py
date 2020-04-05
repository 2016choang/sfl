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
        h, w, c = image_shape  # 19 x 19 x 3

        self.output_size = output_size
        self.feature_size = feature_size

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, (3, 3), stride=2),  # 9 x 9 x 16
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=1),  # 7 x 7 x 16
            nn.ReLU(),
            nn.Flatten(),  # 784
            nn.Linear(784, self.feature_size)  # feature_size 
        )

        self.inverse = nn.Sequential(
            nn.Linear(feature_size * 2, self.output_size)
        )

    def forward(self, obs, next_obs=None, mode='inverse'):
        x = obs.type(torch.float)
        x = x.permute(0, 3, 1, 2)

        if mode == 'inverse':
            next_x = next_obs.type(torch.float)
            next_x = next_x.permute(0, 3, 1, 2)
            embedding = self.encoder(x)
            next_embedding = self.encoder(next_x)
            return self.inverse(torch.cat((embedding, next_embedding), dim=1))
        elif mode == 'encode':
            return self.encoder(x)
        else:
            raise ValueError('Invalid mode!')