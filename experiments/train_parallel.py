
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import json

import torch

from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.dqn.dsr.dsr import DSR
from rlpyt.algos.dqn.dsr.idf_dsr import IDFDSR
from rlpyt.algos.dqn.dsr.action_dsr import ActionDSR
from rlpyt.algos.dqn.dsr.tabular_dsr import TabularDSR
from rlpyt.agents.dqn.dsr.grid_dsr_agent import GridDsrAgent
from rlpyt.agents.dqn.dsr.idf_dsr_agent import IDFDSRAgent
from rlpyt.agents.dqn.dsr.tabular_dsr_agent import TabularFeaturesDsrAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.seed import set_seed


def build_and_train(config_file,
                    env_id="MiniGrid-FourRooms-v0",
                    run_ID=0,
                    cuda_idx=None,
                    workers=2,
                    snapshot_gap=5000,
                    steps=2e4,
                    checkpoint=None):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except ValueError:
        raise ValueError('Unable to read config file {}'.format(config_file))

    mode = config['mode']
    tabular = config['tabular']
    seed = config['seed']
    set_seed(seed)

    if cuda_idx is not None:
        device = torch.device('cuda', index=cuda_idx)
    else:
        device = torch.device('cpu')

    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id, mode=mode, minigrid_config=config['env']),
        CollectorCls=GpuWaitResetCollector,
        eval_env_kwargs=dict(id=env_id, mode=mode, minigrid_config=config['env']),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=32,  # One environment (i.e. sampler Batch dimension).
        eval_n_envs=10,
        eval_max_steps=int(1e3),
        eval_max_trajectories=5,
    )

    if checkpoint is not None:
        if tabular:
            model_checkpoint = torch.load(checkpoint, map_location=device)['agent_state_dict']
        else:
            agent_state_dict = torch.load(checkpoint, map_location=device)['agent_state_dict']
            if mode == 'image':
                idf_model_checkpoint = agent_state_dict['idf_model']
            model_checkpoint = agent_state_dict['model']
    else:
        model_checkpoint = None
        idf_model_checkpoint = None

    if tabular:
        agent = TabularFeaturesDsrAgent(initial_M=model_checkpoint, **config['agent'])
        algo = TabularDSR(**config['algo'])
    else:  
        if mode == 'image':
            agent = IDFDSRAgent(initial_model_state_dict=model_checkpoint,
                                initial_idf_model_state_dict=idf_model_checkpoint, **config['agent'])
            algo = IDFDSR(**config['algo'])
        else:
            agent = GridDsrAgent(mode=mode,initial_model_state_dict=model_checkpoint, **config['agent'])
            algo = DSR(**config['algo'])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=steps,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(workers))),
        seed=seed
    )
    config['env_id'] = env_id
    name = "dsr_" + env_id
    log_dir = mode
    with logger_context(log_dir, run_ID, name, config, snapshot_mode='gap', snapshot_gap=snapshot_gap, tensorboard=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='config file')
    parser.add_argument('--env_id', help='environment ID', default='MiniGrid-FourRooms-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--workers', help='number of sampler workers', type=int, default=2)
    parser.add_argument('--snapshot_gap', help='iterations between snapshots', type=int, default=5000)
    parser.add_argument('--steps', help='iterations', type=float, default=2e4)
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    args = parser.parse_args()
    build_and_train(
        config_file=args.config,
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        workers=args.workers,
        snapshot_gap=args.snapshot_gap,
        steps=args.steps,
        checkpoint=args.checkpoint
    )

