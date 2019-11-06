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

        h, w, c = image_shape  # 152 x 152 x 3

        # Want feature encoding of 256 (4 * 4 * 16)
        self.image_embedding_size = 256

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 6, (4, 4), stride=2), # 75 x 75 x 6
            nn.LeakyReLU(),
            nn.Conv2d(6, 8, (3, 3), stride=2), # 37 x 37 x 8
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (3, 3), stride=2), # 18 x 18 x 16
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, (4, 4), stride=2), # 8 x 8 x 16
            nn.LeakyReLU(),
            nn.Flatten(),  # 1024
            nn.Linear(1024, self.image_embedding_size)  # 256
        )

        self.fc_deconv = nn.Sequential(
            nn.Linear(self.image_embedding_size, 1024),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2), # 18 x 18 x 16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2), # 37 x 37 x 8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 6, kernel_size=3, stride=2), # 75 x 75 x 6
            nn.LeakyReLU(),
            nn.ConvTranspose2d(6, c, kernel_size=4, stride=2), # 152 x 152 x 3
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
                x = self.fc_deconv(features)
                x = x.view(T * B, 16, 8, 8)
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


class GridDsrSmallModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            ):
        super().__init__()
        self.output_size = output_size

        h, w, c = image_shape  # 84 x 84 x 3

        # Want feature encoding of 512 (4 * 4 * 32)
        self.image_embedding_size = 512

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 4, (4, 4), stride=2), # 41 x 41 x 4
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, (3, 3), stride=2), # 20 x 20 x 8
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (4, 4), stride=2), # 9 x 9 x 16
            nn.LeakyReLU(),
            nn.Flatten(),  # 1296
            nn.Linear(1296, self.image_embedding_size)  # 512
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2), # 9 x 9 x 16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2), # 20 x 20 x 8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2), # 41 x 41 x 4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, c, kernel_size=4, stride=2), # 84 x 84 x 3
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
                x = x.view(T * B, 32, 4, 4)
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


class GridDsrCompactModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=256,
            ):
        super().__init__()
        self.output_size = output_size

        h, w, c = image_shape  # 19 x 19 x 3

        # Want feature encoding of 32 (2 x 2 x 8)
        self.image_embedding_size = 32

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 6, (3, 3), stride=2), # 9 x 9 x 6 
            nn.LeakyReLU(),
            nn.Flatten(),  # 486
            nn.Linear(486, self.image_embedding_size)  # 32
        )

        self.fc_deconv = nn.Sequential(
            nn.Linear(self.image_embedding_size, 486),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(6, c, kernel_size=3, stride=2), # 19 x 19 x 3
        )

        self.dsr = MlpModel(self.image_embedding_size, fc_sizes,
            output_size=self.image_embedding_size * output_size)

    def forward(self, x, mode='features'):
        if mode == 'features' or mode =='reconstruct':
            x = x.type(torch.float)
            # x = (x - x.mean(dim=[0, 1, 2])) / x.std(dim=[0, 1, 2])  # compact does not need to normalize
            x = x.permute(0, 3, 1, 2)
            lead_dim, T, B, img_shape = infer_leading_dims(x, 3)

            x = self.encoder(x.view(T * B, *img_shape))
            features = x.view(T * B, -1)

            if mode == 'reconstruct':
                x = self.fc_deconv(features)
                x = x.view(T * B, 6, 9, 9)
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