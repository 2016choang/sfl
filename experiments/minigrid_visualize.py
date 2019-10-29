import pickle

import gym
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper
import numpy as np
import torch

from rlpyt.models.dqn.grid_dsr_model import GridDsrModel


ENV_ID = 'MiniGrid-FourRooms-v0'


def visualize(checkpoint, output, cuda_idx=None):
    if cuda_idx is not None:
        device = torch.device('cuda', index=cuda_idx)
    else:
        device = torch.device('cpu')

    # load in checkpoint into agent
    params = torch.load(checkpoint, map_location=device)

    
    # sample all possible agent positions within environment
    env = RGBImgObsWrapper(ReseedWrapper(gym.make(id=ENV_ID)))
    observation = env.reset()

    starting_pos = tuple(env.unwrapped.agent_pos)
    print(starting_pos)

    SR = {}

    model = GridDsrModel(env.observation_space.shape, env.action_space.n)
    model.load_state_dict(params['agent_state_dict']['model'])
    model.to(device)

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
            SR[pos] = dsr
            unique_states += 1
            if unique_states == 210:
                break

        if (step + 1) % 1000 == 0:
            print('Completed {} steps'.format(step + 1))

        observation, _, _, _ = env.step(env.action_space.sample())

    env.close()

    with open(output, 'wb') as handle:
        pickle.dump(SR, handle)

    # calculate L2 norm between SR of beginning state and other position



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='checkpoint file')
    parser.add_argument('--output', help='output location')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    visualize(checkpoint=args.input, output=args.output, cuda_idx=args.cuda_idx)