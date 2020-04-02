import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


FUNCTION_MAP = {'identity': nn.Identity,
                'relu': nn.ReLU,
                'leaky': nn.LeakyReLU}


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)  # B x self.shape...


class InverseDynamicsModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=512
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

    def forward(self, obs, next_obs, mode='inverse'):
        x = obs.type(torch.float)
        next_x = next_obs.type(torch.float)
        x = x.permute(0, 3, 1, 2)
        next_x = next_x.permute(0, 3, 1, 2)
        if mode == 'inverse':
            embedding = self.encoder(x)
            next_embedding = self.encoder(next_x)
            return self.inverse(torch.cat((embedding, next_embedding), dim=1))
        elif mode == 'encode':
            return self.encoder(x), self.encoder(next_x)
        else:
            raise ValueError('Invalid mode!')


class ForwardPixelModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=512
            ):
        super().__init__()
        h, w, c = image_shape  # 19 x 19 x 3

        self.onehot = torch.eye(output_size).float()

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

        self.dynamics = nn.Sequential(
            nn.Linear(self.feature_size + self.output_size, 784),
            nn.ReLU(),
            Reshape(16, 7, 7),
            nn.ConvTranspose2d(16, 16, (3, 3), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, (3, 3), stride=2)
        )

    def forward(self, obs, action, mode='dyamics'):
        x = obs.type(torch.float)
        x = x.permute(0, 3, 1, 2)
        act = self.onehot[action].to(action.device)
        if mode == 'dynamics':
            embedding = self.encoder(x)
            return self.dynamics(torch.cat((embedding, act), dim=1)).permute(0, 2, 3, 1)
        elif mode == 'encode':
            return self.encoder(x)
        else:
            raise ValueError('Invalid mode!')


class ForwardFeatureModel(torch.nn.Module):

    def __init__(
            self,
            output_size,
            feature_size=512,
            fc_sizes=[512]
            ):
        super().__init__()
        self.output_size = output_size
        self.feature_size = feature_size

        self.dynamics = MlpModel(self.feature_size + self.output_size, fc_sizes,
            output_size=self.feature_size, nonlinearity=nn.ReLU)

        self.onehot = torch.eye(output_size).float()

    def forward(self, obs, action):
        x = obs.type(torch.float)
        act = self.onehot[action].to(action.device)
        return self.dynamics(torch.cat((x, act), dim=1))


class FeatureDSRModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=512,
            dynamics_params={},
            dsr_params={}
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

        self.dynamics = MlpModel(self.feature_size + self.output_size, dsr_params['fc_sizes']['fc_sizes'],
            output_size=self.feature_size, nonlinearity=FUNCTION_MAP[dynamics_params['nonlinearity']])

        self.onehot = torch.eye(output_size).float()

        self.dsr = MlpModel(self.feature_size, dsr_params['fc_sizes'],
            output_size=self.feature_size, nonlinearity=FUNCTION_MAP[dsr_params['nonlinearity']])

        self.q_estimate = nn.Linear(self.feature_size, 1)

    def forward(self, x, next_x=None, action=None, mode=None):
        if mode == 'inverse':
            return self.inverse(torch.cat((x, next_x), dim=1))
        elif mode == 'encode':
            x = x.type(torch.float).permute(0, 3, 1, 2)
            next_x = next_x.type(torch.float).permute(0, 3, 1, 2)
            return self.encoder(x), self.encoder(next_x)
        elif mode == 'dynamics':
            act = self.onehot[action].to(action.device)
            return self.dynamics(torch.cat((x, act), dim=1))
        elif mode == 'dsr':
            return self.dsr(x)
        else:
            raise ValueError('Invalid mode!')


class GridReconstructModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            feature_size=512,
            dsr_params={}
            ):
        super().__init__()
        h, w, c = image_shape  # 84 x 84 x 3

        self.output_size = output_size

        self.feature_size = feature_size

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 8, (4, 4), stride=2),  # 41 x 41 x 8
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), stride=2),  # 20 x 20 x 16
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2),  # 9 x 9 x 32
            nn.ReLU(),
            nn.Flatten(),  # 2592
            nn.Linear(2592, self.feature_size),  # feature_size 
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.feature_size, 2592),  # 2592
            nn.ReLU(),
            Reshape(32, 9, 9),  # 9 x 9 x 32
            nn.ConvTranspose2d(32, 16, (4, 4), stride=2), # 20 x 20 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (3, 3), stride=2), # 41 x 41 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(8, c, (4, 4), stride=2), # 84 x 84 x 3
        )

        self.dsr = MlpModel(self.feature_size, dsr_params['fc_sizes'],
            output_size=self.output_size * self.feature_size, nonlinearity=FUNCTION_MAP[dsr_params['nonlinearity']])

        self.q_estimate = nn.Linear(self.feature_size, 1)

    def forward(self, observation, mode='encode'):
        x = observation.type(torch.float)
        if mode == 'encode':
            x = x.permute(0, 3, 1, 2)
            return self.encoder(x)
        elif mode == 'decode':
            decoded = self.decoder(x)
            return decoded.permute(0, 2, 3, 1)
        elif mode == 'dsr':
            dsr = self.dsr(x)
            return dsr.reshape(-1, self.output_size, self.feature_size)
        elif mode == 'q':
            q = self.q_estimate(x).squeeze(2)
            return q
        else:
            raise ValueError('Invalid mode!')


class GridGoalModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            dsr_params={},
            goal_params={}
            ):
        super().__init__()
        self.feature_size = image_shape[0]
        self.output_size = output_size

        self.dsr = MlpModel(self.feature_size, dsr_params['fc_sizes'],
            output_size=self.output_size * self.feature_size, nonlinearity=FUNCTION_MAP[dsr_params['nonlinearity']])

        self.goal = MlpModel(self.feature_size, goal_params['fc_sizes'],
            output_size=self.feature_size, nonlinearity=FUNCTION_MAP[goal_params['nonlinearity']])

        self.q_estimate = nn.Linear(self.feature_size, 1)

    def forward(self, observation, mode='encode'):
        x = observation.type(torch.float)
        if mode == 'encode':
            return x
        elif mode == 'reconstruct':
            return x
        elif mode == 'goal':
            goal_embedding = self.goal(x)
            return goal_embedding / torch.norm(goal_embedding, p=2, dim=1, keepdim=True)
        elif mode == 'dsr':
            dsr = self.dsr(x)
            return dsr.reshape(-1, self.output_size, self.feature_size)
        elif mode == 'q':
            q = self.q_estimate(x).squeeze(2)
            return q
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


class GridDsrModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            dsr_params={}
            ):
        super().__init__()
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


class GridDsrFullModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            ):
        super().__init__()
        self.output_size = output_size

        h, w, c = image_shape  # 152 x 152 x 3

        # Want feature encoding of 512 (4 * 4 * 32)
        self.image_embedding_size = 512

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 6, (4, 4), stride=2), # 75 x 75 x 6
            nn.LeakyReLU(),
            nn.Conv2d(6, 8, (3, 3), stride=2), # 37 x 37 x 8
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (3, 3), stride=2), # 18 x 18 x 16
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=2), # 8 x 8 x 32
            nn.LeakyReLU(),
            nn.Flatten(),  # 2048
            nn.Linear(2048, self.image_embedding_size)  # 512
        )

        self.fc_deconv = nn.Sequential(
            nn.Linear(self.image_embedding_size, 2048),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2), # 18 x 18 x 16
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
            x = x.permute(0, 3, 1, 2)
            lead_dim, T, B, img_shape = infer_leading_dims(x, 3)

            x = self.encoder(x.view(T * B, *img_shape))
            features = x.view(T * B, -1)

            if mode == 'reconstruct':
                x = self.fc_deconv(features)
                x = x.view(T * B, 32, 8, 8)
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

        # Want feature encoding of 1204
        self.image_embedding_size = 1024

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 64, (6, 6), stride=2), # 40 x 40 x 64
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (6, 6), padding=2, stride=2), # 20 x 20 x 64
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (6, 6), padding=2, stride=2), # 10 x 10 x 64
            nn.LeakyReLU(),
            nn.Flatten(),  # 6400
            nn.Linear(6400, self.image_embedding_size),  # 1024
            nn.LeakyReLU()
        )

        self.fc_deconv = nn.Sequential(
            nn.Linear(self.image_embedding_size, 6400),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=6, padding=2, stride=2), # 20 x 20 x 64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=6, padding=2, stride=2), # 40 x 40 x 64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, c, kernel_size=6, stride=2), # 84 x 84 x 3
        )

        self.dsr = MlpModel(self.image_embedding_size, fc_sizes,
            output_size=self.image_embedding_size * output_size)

    def forward(self, x, mode='features'):
        if mode == 'features' or mode =='reconstruct':
            x = x.type(torch.float)
            x = x.permute(0, 3, 1, 2)
            lead_dim, T, B, img_shape = infer_leading_dims(x, 3)

            x = self.encoder(x.view(T * B, *img_shape))
            features = x.view(T * B, -1)

            if mode == 'reconstruct':
                x = self.fc_deconv(features)
                x = x.view(T * B, 64, 10, 10)
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
