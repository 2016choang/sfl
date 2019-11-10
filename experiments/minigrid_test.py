
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.dqn.dsr.dsr import DSR
from rlpyt.agents.dqn.grid_dsr.grid_dsr_agent import GridDsrAgent, GridDsrSmallAgent, GridDsrCompactAgent, GridDsrRandomAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.seed import set_seed


def build_and_train(env_id="MiniGrid-FourRooms-v0", run_ID=0, cuda_idx=None, mode='full', seed=333, snapshot_gap=5000):
    set_seed(seed)

    minigrid_config = {'mode': mode}

    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id, minigrid_config=minigrid_config),
        eval_env_kwargs=dict(id=env_id, minigrid_config=minigrid_config),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    # lr_schedule_config={'mode': 'plateau',
    #                     'patience': 0}
    learn_re = mode != 'random'

    algo = DSR(batch_size=32,
               min_steps_learn=int(1e3),
               learning_rate=1e-5,
               replay_size=int(1e5),
               learn_re=learn_re
               )
    if mode == 'full':
        agent = GridDsrAgent()
    elif mode == 'small':
        agent = GridDsrSmallAgent()
    elif mode == 'compact':
        agent = GridDsrCompactAgent()
    elif mode == 'random':
        agent = GridDsrRandomAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=2e4,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
        seed=seed
    )
    config = dict(env_id=env_id)
    name = "dsr_" + env_id
    log_dir = "minigrid_test"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode='gap', snapshot_gap=snapshot_gap, tensorboard=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='MiniGrid-FourRooms-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--mode', help='full, small, compact, random', choices=['full', 'small', 'compact', 'random'])
    parser.add_argument('--seed', help='seed', type=int, default=333)
    parser.add_argument('--snapshot_gap', help='iterations between snapshots', type=int, default=5000)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        mode=args.mode,
        seed=args.seed,
        snapshot_gap=args.snapshot_gap
    )
