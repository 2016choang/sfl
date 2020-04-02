import json
import pickle

import gym
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper
import numpy as np
import torch

from rlpyt.envs.gym import make as gym_make
from rlpyt.models.dqn.grid_dsr_model import GridActionDsrModel
from rlpyt.utils.seed import set_seed


ENV_ID = 'MiniGrid-FourRooms-v0'

def test(config_file, 
              checkpoint,
              output,
              cuda_idx=None):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except ValueError:
        raise ValueError('Unable to read config file {}'.format(config_file))

    mode = config['mode']
    seed = config['seed']
    set_seed(seed)

    if cuda_idx is not None:
        device = torch.device('cuda', index=cuda_idx)
    else:
        device = torch.device('cpu')

    # load in checkpoint into agent
    params = torch.load(checkpoint, map_location=device)

    # sample all possible agent positions within environment
    env = gym_make(id=ENV_ID, mode=mode, minigrid_config=config['env'])
    env.reset()
    
    SR = torch.zeros((19, 19, env.action_space.n, env.observation_space.shape[0]), dtype=torch.float)
    SR += np.nan

    model = GridActionDsrModel(env.observation_space.shape, env.action_space.n, **config['agent']['model_kwargs'])
    model.load_state_dict(params['agent_state_dict']['model'])
    model.to(device)

    positions = set()
    for y in range(19):
        for x in range(19):
            if x not in [0, 9, 18] and y not in [0, 9, 18]:
                for action in range(4):
                    env.env.env.unwrapped.agent_pos = np.array([y, x])
                    _, _, done, _ = env.step(action)

                    pos = tuple(env.agent_pos)
                    if pos not in positions:
                        positions.add(pos)

                    if done:
                        env.reset()

    print(len(pos))

    for pos in positions:
        for a in range(4):
            env.env.env.unwrapped.agent_pos = np.array(pos)
            obs, _, done, _ = env.step(4)
            obs = torch.Tensor(obs[a]).unsqueeze(0).to(device)

            features = model(obs)
            SR[pos[0], pos[1], a] = model(features, 'dsr')

            if done:
                env.reset()

    env.close()

    torch.save(SR, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--output', help='output location')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    test(config_file=args.config,
              checkpoint=args.checkpoint,
              output=args.output, 
              cuda_idx=args.cuda_idx)