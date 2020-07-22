import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import FUNCTION_MAP, Reshape


class DsrModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=None,
            dsr_params={}
            ):
        super().__init__()
        if feature_size is not None:
            self.feature_size = feature_size
        else:
            self.feature_size = image_shape[0]
        self.output_size = output_size

        self.dsr = MlpModel(self.feature_size, dsr_params['fc_sizes'],
            output_size=self.output_size * self.feature_size, nonlinearity=FUNCTION_MAP[dsr_params['nonlinearity']])

        self.q_estimate = nn.Linear(self.feature_size, 1)

    def forward(self, observation, mode='encode'):
        x = observation.type(torch.float)
        if mode == 'encode':
            return x
        elif mode == 'dsr':
            dsr = self.dsr(x)
            return dsr.reshape(-1, self.output_size, self.feature_size)
        elif mode == 'q':
            q = self.q_estimate(x).squeeze(2)
            return q
        else:
            raise ValueError('Invalid mode!')


class GridActionDsrModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            dsr_params={}
            ):
        super().__init__()
        self.feature_size = image_shape[1]
        self.output_size = output_size

        self.dsr = MlpModel(self.feature_size, dsr_params['fc_sizes'],
            output_size=self.feature_size, nonlinearity=FUNCTION_MAP[dsr_params['nonlinearity']])

    def forward(self, observation, mode='encode'):
        x = observation.type(torch.float)
        if mode == 'encode':
            return x
        elif mode == 'dsr':
            return self.dsr(x)
        elif mode == 'q':
            return torch.zeros(x.shape[:2], device=x.device)
        else:
            raise ValueError('Invalid mode!')


class GridDsrDummyModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            dsr_params={}
            ):
        super().__init__()
