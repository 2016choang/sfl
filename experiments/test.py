import json
import pickle

import gym
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper
import numpy as np
import torch

from rlpyt.envs.gym import make as gym_make
from rlpyt.models.dqn.dsr.grid_dsr_model import GridDsrModel
from rlpyt.models.dqn.dsr.idf_model import IDFModel
from rlpyt.utils.seed import set_seed


ENV_ID = 'MiniGrid-FourRooms-v0'

def visualize(config_file, 
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

    # starting_pos = tuple(env.unwrapped.agent_pos)
    # print(starting_pos)
    
    SR = torch.zeros((19, 19, env.action_space.n, config['agent']['model_kwargs']['feature_size']),
                     dtype=torch.float)
    SR += np.nan
    seen = set()

    if mode == 'image':
        feature_model = IDFModel(env.observation_space.shape, env.action_space.n, **config['agent']['idf_model_kwargs'])
        feature_model.load_state_dict(params['agent_state_dict']['idf_model'])
        feature_model.to(device)
    
    model = GridDsrModel(env.observation_space.shape, env.action_space.n, **config['agent']['model_kwargs'])
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
                        features = feature_model(obs.to(device), mode='encode')

                    sr_y, sr_x = tuple(env.agent_pos)
                    SR[sr_y, sr_x] = model(features, mode='dsr')

                    if done:
                        env.reset()

    print(len(seen))
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
    visualize(config_file=args.config,
              checkpoint=args.checkpoint,
              output=args.output, 
              cuda_idx=args.cuda_idx)