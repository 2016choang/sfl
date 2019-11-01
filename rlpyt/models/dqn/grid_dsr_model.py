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

        h, w, c = image_shape  # 84 x 84 x 3

        # Want feature encoding of 256 (4 * 4 * 16)
        self.image_embedding_size = 256

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 4, (3, 3), stride=3), # 28 x 28 x 4
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, (3, 3), padding=1, stride=3), # 10 x 10 x 8
            nn.LeakyReLU(),
            nn.Flatten(),  # 800
            nn.Linear(800, self.image_embedding_size)  # 256
            # nn.Conv2d(8, 8, (4, 4))  # 4 x 4 x 8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2), # 9 x 9 x 8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=3), # 27 x 27 x 4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, c, kernel_size=6, stride=3), # 84 x 84 x 3
        )

        self.dsr = MlpModel(self.image_embedding_size, fc_sizes,
            output_size=self.image_embedding_size * output_size)

    def forward(self, x, mode='features'):
        if mode == 'features' or mode =='reconstruct':
            x = x.type(torch.float)
            x = (x - x.mean(dim=[0, 1, 2])) / x.std(dim=[0, 1, 2])
            x = x.permute(0, 3, 1, 2)
            lead_dim, T, B, img_shape = infer_leading_dims(x, 3)

            x = self.encoder(x.view(T * B, *img_shape))
            features = x.view(T * B, -1)

            if mode == 'reconstruct':
                x = x.view(T * B, 16, 4, 4)
                reconstructed = self.decoder(x).permute(0, 2, 3, 1)
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
