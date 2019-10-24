import pickle

import gym
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper
import numpy as np
import torch

from rlpyt.models.dqn.grid_dsr_model import GridDsrModel


ENV_ID = 'MiniGrid-FourRooms-v0'


def visualize(checkpoint, cuda_idx):
    # device = torch.device('cuda', index=cuda_idx)

    # load in checkpoint into agent
    params = torch.load(checkpoint, map_location=torch.device('cpu'))

    
    # sample all possible agent positions within environment
    env = RGBImgObsWrapper(ReseedWrapper(gym.make(id=ENV_ID)))
    observation = env.reset()

    starting_pos = tuple(env.unwrapped.agent_pos)

    SR = {}

    model = GridDsrModel(env.observation_space.shape, env.action_space.n)
    model.load_state_dict(params['agent_state_dict']['model'])
    # model.to(device)

    unique_states = 0
    step = 0

    while True:
        pos = tuple(env.unwrapped.agent_pos)
        observation = torch.Tensor(observation).unsqueeze(0)

        # calculate successor reprsentation for each position
        if pos not in SR:
            with torch.no_grad():
                # features = model(observation.to(device))
                features = model(observation)

            dsr = model(features, mode='dsr').mean(dim=1).squeeze(0)
            print(model(features, mode='dsr').std())
            SR[pos] = dsr
            unique_states += 1
            print(unique_states)
            if unique_states == 250:
                break

        if (step + 1) % 1000 == 0:
            print('Completed {} steps'.format(step + 1))

        observation, _, _, _ = env.step(env.action_space.sample())

    env.close()

    with open('successor.pkl', 'wb') as handle:
        pickle.dump(SR, handle)

    # calculate L2 norm between SR of beginning state and other position



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', help='checkpoint file')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    visualize(checkpoint=args.file, cuda_idx=args.cuda_idx)