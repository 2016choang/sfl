import pickle

import gym
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper
import numpy as np
import torch

from rlpyt.envs.gym import make as gym_make
from rlpyt.models.dqn.grid_dsr_model import GridDsrModel, GridDsrSmallModel, GridDsrCompactModel, GridDsrRandomModel
from rlpyt.utils.seed import set_seed


ENV_ID = 'MiniGrid-FourRooms-v0'


def visualize(checkpoint, output, cuda_idx=None, mode='full', seed=333):
    set_seed(seed)

    if cuda_idx is not None:
        device = torch.device('cuda', index=cuda_idx)
    else:
        device = torch.device('cpu')

    # load in checkpoint into agent
    params = torch.load(checkpoint, map_location=device)

    
    # sample all possible agent positions within environment
    minigrid_config = {'mode': mode}
    env = gym_make(id=ENV_ID, minigrid_config=minigrid_config)
    env.reset()

    # starting_pos = tuple(env.unwrapped.agent_pos)
    # print(starting_pos)

    SR = torch.zeros((19, 19, env.action_space.n, env.observation_space.shape), dtype=torch.float)
    seen = set()

    if mode == 'full':
        model = GridDsrModel(env.observation_space.shape, env.action_space.n)
    elif mode == 'small':
        model = GridDsrSmallModel(env.observation_space.shape, env.action_space.n)
    elif mode == 'compact':
        model = GridDsrCompactModel(env.observation_space.shape, env.action_space.n)
    elif mode == 'random':
        model = GridDsrRandomModel(env.observation_space.shape, env.action_space.n)
    model.load_state_dict(params['agent_state_dict']['model'])
    model.to(device)

    for y in range(19):
        for x in range(19):
            if x not in [0, 9, 18] and y not in [0, 9, 18]:
                env.env.env.unwrapped.agent_pos = np.array([y, x])
                for a in range(env.action_space.n):
                    obs, _, _, _ = env.step(a)
                    seen.add(tuple(env.agent_pos))
                    env.step((a + 2) % env.action_space.n)

                    with torch.no_grad():
                        features = model(obs.to(device))
                    act = torch.zeros(env.action_space.n, dtype=torch.float).unsqueeze(0).to(device)
                    act[0, a] = 1
                    SR[y, x, a] = model(features, act, mode='dsr')

    print(len(seen))
    env.close()

    torch.save(SR, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='checkpoint file')
    parser.add_argument('--output', help='output location')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--mode', help='full, small, compact, random', choices=['full', 'small', 'compact', 'random'])
    parser.add_argument('--seed', help='seed', type=int, default=333)
    parser.add_argument('--')
    args = parser.parse_args()
    visualize(checkpoint=args.input,
              output=args.output, 
              cuda_idx=args.cuda_idx,
              mode=args.mode,
              seed=args.seed)