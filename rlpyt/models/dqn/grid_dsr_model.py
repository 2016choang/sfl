import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class GridDsrModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            ):
        super().__init__()
        self.output_size = output_size

        h, w, c = image_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2)),
            nn.ReLU()
        )

        self.image_embedding_size = (((h - 1) // 2 - 1) // 2 - 2) * (((w - 1) // 2 - 1) // 2 - 2) * 64

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (5, 5)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, c, (5, 5)),
            nn.Tanh()
        )

        self.dsr = MlpModel(self.image_embedding_size, fc_sizes,
            output_size=self.image_embedding_size * output_size)

    def forward(self, x, mode='features'):
        if mode == 'features' or mode =='reconstruct':
            x = x.transpose(1, 3).transpose(2, 3).type(torch.float)
            lead_dim, T, B, img_shape = infer_leading_dims(x, 3)

            x = self.encoder(x.view(T * B, *img_shape))
            features = x.view(T * B, -1)
            
            if mode == 'reconstruct':
                reconstructed = self.decoder(x).transpose(3, 1).transpose(2, 1)
                reconstructed = restore_leading_dims(reconstructed, lead_dim, T, B)
                return reconstructed
            else:
                features = restore_leading_dims(features, lead_dim, T, B)
                return features

        elif mode == 'dsr':
            lead_dim, T, B, img_shape = infer_leading_dims(x, 1)
            dsr = self.dsr(x)
            dsr = restore_leading_dims(dsr, lead_dim, T, B).view(-1, self.output_size, *img_shape)
            return dsr
        else:
            raise ValueError('Invalid mode!')
