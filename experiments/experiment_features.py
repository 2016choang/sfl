import os

import keras
import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter

from rlpyt.models.dqn.dsr.tcf_model import FixedVizDoomModel
from rlpyt.utils.seed import set_seed


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def build_and_train(config_file,
                    run_ID,
                    cuda_idx,
                    snapshot_gap,
                    steps,
                    checkpoint,
                    logdir):

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

    logdir = os.path.join(logdir, 'run_{}'.format(run_ID))
    writer = SummaryWriter(logdir=logdir)

    epochs = config['epochs']
    batch_size = config['batch_size']

    for epoch in range(epochs):
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='config file')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--snapshot_gap', help='iterations between snapshots', type=int, default=10000)
    parser.add_argument('--steps', help='iterations', type=float, default=200000)
    parser.add_argument('--logdir', help='log directory', default='data')
    args = parser.parse_args()
    build_and_train(
        config_file=args.config,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        snapshot_gap=args.snapshot_gap,
        steps=args.steps,
        logdir=args.logdir
    )