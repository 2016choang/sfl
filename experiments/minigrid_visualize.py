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

    SR = torch.zeros((19, 19, env.action_space.n, env.observation_space.shape[0]), dtype=torch.float)
    SR += np.nan
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
                for a in range(env.action_space.n):
                    env.env.env.unwrapped.agent_pos = np.array([y, x])
                    obs, _, done, _ = env.step(a)

                    obs = torch.Tensor(obs).unsqueeze(0)
                    seen.add(tuple(env.agent_pos))

                    with torch.no_grad():
                        features = model(obs.to(device))

                    for i in range(env.action_space.n):
                        act = torch.zeros(env.action_space.n, dtype=torch.float).unsqueeze(0).to(device)
                        act[0, i] = 1
                        sr_y, sr_x = tuple(env.agent_pos)
                        SR[sr_y, sr_x, i] = model(features, act, mode='dsr')

                    if done:
                        env.reset()

    print(len(seen))
    env.close()

    torch.save(SR, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='checkpoint file')
    parser.add_argument('--output', help='output location')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--mode', help='full, small, compact, random', choices=['full', 'small', 'compact', 'random'], default='random')
    parser.add_argument('--seed', help='seed', type=int, default=333)
    args = parser.parse_args()
    visualize(checkpoint=args.input,
              output=args.output, 
              cuda_idx=args.cuda_idx,
              mode=args.mode,
              seed=args.seed)