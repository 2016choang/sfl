"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import copy
import json
import os

import keras
import tensorflow as tf
import torch

from rlpyt.samplers.serial.sampler import VizdoomSampler
from rlpyt.samplers.serial.collectors import SerialVizdoomEvalCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuLandmarksCollector
from rlpyt.envs.vizdoom.vizdoom_env import VizDoomEnv
from rlpyt.algos.dqn.dsr.dsr import DSR
from rlpyt.algos.dqn.dsr.feature_dsr import IDFDSR, LandmarkTCFDSR, FixedFeatureDSR, PositionPrediction
from rlpyt.agents.dqn.dsr.landmark_agent import LandmarkVizDoomAgent 
from rlpyt.agents.dqn.dsr.landmarks import Landmarks
from rlpyt.models.dqn.dsr.tcf_model import VizDoomTCFModel, FixedVizDoomModel
from rlpyt.runners.minibatch_rl import MinibatchVizDoomLandmarkDSREval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.seed import set_seed


def build_and_train(config_file,
                    run_ID=0,
                    cuda_idx=None,
                    snapshot_gap=5000,
                    steps=2e4,
                    checkpoint=None):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except ValueError:
        raise ValueError('Unable to read config file {}'.format(config_file))

    seed = config['seed']
    set_seed(seed)

    if cuda_idx is not None:
        device = torch.device('cuda', index=cuda_idx)
    else:
        device = torch.device('cpu')

    sampler = VizdoomSampler(
        CollectorCls=CpuLandmarksCollector,
        eval_CollectorCls=SerialVizdoomEvalCollector,
        EnvCls=VizDoomEnv,
        env_kwargs=config['env'],
        eval_env_kwargs=config['eval_env'],
        batch_T=1,  # One time-step per sampler iteration.
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(2e4),
        eval_max_trajectories=1,
        **config['sampler']
    )    

    if checkpoint is not None:
        agent_state_dict = torch.load(checkpoint, map_location=device)['agent_state_dict']
        feature_model_checkpoint = agent_state_dict['feature_model']
        model_checkpoint = agent_state_dict['model']
    else:
        model_checkpoint = None
        feature_model_checkpoint = None

    feature = config['feature']
    agent_class = LandmarkVizDoomAgent
    if feature == 'TCF':
        # featureModelCls = VizDoomTCFModel
        featureModelCls = FixedVizDoomModel
        # algo_class = LandmarkTCFDSR
        algo_class = FixedFeatureDSR
        # algo_class = PositionPrediction
    else:
        raise NotImplementedError

    landmarks = Landmarks(**config['landmarks'])
    if 'landmarks_path' in config:
        landmarks.load(config['landmarks_path'], device)
    agent = agent_class(featureModelCls=featureModelCls,
                        initial_model_state_dict=model_checkpoint, 
                        initial_feature_model_state_dict=feature_model_checkpoint,
                        landmarks=landmarks,
                        **config['agent'])
    algo = algo_class(**config['algo'])
    runner = MinibatchVizDoomLandmarkDSREval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=steps,
        affinity=dict(cuda_idx=cuda_idx),
        seed=seed,
        **config['runner']
    )
    name = "dsr_vizdoom"
    log_dir = "vizdoom"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode='gap', snapshot_gap=snapshot_gap, tensorboard=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='config file')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--snapshot_gap', help='iterations between snapshots', type=int, default=10000)
    parser.add_argument('--steps', help='iterations', type=float, default=200000)
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--gpu_fraction', help='gpu fraction', type=float, default=0.3)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_idx)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    build_and_train(
        config_file=args.config,
        run_ID=args.run_ID,
        cuda_idx=0,
        snapshot_gap=args.snapshot_gap,
        steps=args.steps,
        checkpoint=args.checkpoint
    )
