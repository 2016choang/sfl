import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import FUNCTION_MAP, Reshape


class TCFModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=64,
            norm_output=True,
            alpha=10.0,
            initial_stride=2,
            simple_encoder=False
            ):
        super().__init__()
        h, w, c = image_shape

        self.output_size = output_size
        self.feature_size = feature_size
        self.norm_output = norm_output
        self.alpha = alpha

        embedding_c = 32
        embedding_s = ((h - 3) // initial_stride) + 1 - 3 + 1

        conv_embedding_size = embedding_c * embedding_s * embedding_s

        if simple_encoder:
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(h * w * c, 128),
                nn.ReLU(),
                nn.Linear(128, feature_size)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 32, (3, 3), stride=initial_stride),
                nn.ReLU(),
                nn.Conv2d(32, embedding_c, (3, 3), stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(conv_embedding_size, self.feature_size)
            )

    def forward(self, obs, next_obs=None, mode='encode'):
        x = obs.type(torch.float)
        if mode == 'encode':
            x = x.permute(0, 3, 1, 2)
            x = self.encoder(x)
            if self.norm_output:
                return self.normalize(x) * self.alpha
            else:
                return x
        else:
            raise ValueError('Invalid mode!')

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output
