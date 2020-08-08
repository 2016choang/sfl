import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.models.resnet import ResnetBuilder
from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import FUNCTION_MAP, Reshape
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.quick_args import save__init__args


def normalize(x):
    buffer = torch.pow(x, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    normalization_constant = torch.sqrt(normp)
    output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
    return output

class FixedVizDoomModel(torch.nn.Module):

    def __init__(
            self,
            fixed_weights_path,
            feature_size=512,
            final_feature_size=None,
            norm_output=True,
            alpha=10.0,
            **kwargs
        ):
        save__init__args(locals())
        super().__init__()

        self.fixed_model = ResnetBuilder.build_siamese_resnet_18((6, 120, 160), 2)
        self.fixed_model.load_weights(self.fixed_weights_path)
        self.fixed_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        weights = self.fixed_model.layers[5].get_weights()
        self.mean = weights[2][:512]
        self.var = weights[3][:512] ** 2
        self.divisor = np.sqrt(self.var + self.fixed_model.layers[5].epsilon)

        if self.final_feature_size:
            self.encoder = nn.Linear(self.feature_size, self.final_feature_size)
        else:
            self.encoder = nn.Linear(1, 1)
    
    def forward(self, obs, mode='encode'):
        x = obs.type(torch.float)
        if mode == 'encode':
            lead_dim, T, B, img_shape = infer_leading_dims(x, 3)
            x = x.view(T * B, *img_shape).cpu()
            x = x.permute(0, 2, 3, 1)
            x = self.fixed_model.layers[3].predict(x)
            if self.norm_output:
                x = (x - self.mean) / self.divisor
            x = torch.from_numpy(x).to(device=obs.device)
            return restore_leading_dims(x, lead_dim, T, B)
        elif mode == 'output':
            x = self.encoder(x)
            if self.norm_output:
                return normalize(x) * self.alpha
        else:
            raise ValueError('Invalid mode!')
    
    def get_features(self, obs):
        representation = self.forward(obs, mode='encode')
        if self.final_feature_size:
            representation = self.forward(representation, mode='output')
        return representation

class VizDoomTCFModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=512,
            norm_output=True,
            alpha=10.0,
            ):
        super().__init__()
        c, h, w = image_shape

        self.output_size = output_size
        self.feature_size = feature_size
        self.norm_output = norm_output
        self.alpha = alpha

        embedding_c = 64
        embedding_h = 11 #7 # 10
        embedding_w = 16 #7 # 10

        conv_embedding_size = embedding_c * embedding_h * embedding_w

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 64, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), 2),
            nn.ReLU(),
            nn.Conv2d(64, embedding_c, (3, 3), 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_embedding_size, self.feature_size)
        )        

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(c, 64, (6, 6), 2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (6, 6), 2, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (6, 6), 2, padding=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(conv_embedding_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, self.feature_size)
        # )        
        
    def forward(self, obs, mode='encode'):
        x = obs.type(torch.float)
        if mode == 'encode':
            x = self.encoder(x)
            if self.norm_output:
                return normalize(x) * self.alpha
            else:
                return x
        else:
            raise ValueError('Invalid mode!')
    
    def get_features(self, obs):
        return self.forward(obs, mode='encode')

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

    def forward(self, obs, mode='encode'):
        x = obs.type(torch.float)
        if mode == 'encode':
            x = x.permute(0, 3, 1, 2)
            x = self.encoder(x)
            if self.norm_output:
                return normalize(x) * self.alpha
            else:
                return x
        else:
            raise ValueError('Invalid mode!')

    def get_features(self, obs):
        return self.forward(obs, mode='encode')
