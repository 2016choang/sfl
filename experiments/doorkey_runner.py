import argparse
import json
import os
import subprocess
import time


def run(base_file, start_run_id, cuda_idx, snapshot_gap, steps, checkpoint, sleep):
    try:
        with open(base_file, 'r') as f:
            base_config = json.load(f)
    except ValueError:
        raise ValueError('Unable to read base config file {}'.format(base_file))

    run_id = start_run_id
    
    config_dir = os.path.dirname(base_file)
    tmp_config_dir = os.path.join(config_dir, 'tmp')
    if not os.path.exists(tmp_config_dir):
        os.mkdir(tmp_config_dir)
        
    # values = [30, 40, 50]
    values = [0.98, 0.985, 0.99]
    for value in values:
        # base_config['landmarks']['max_landmarks'] = value
        base_config['landmarks']['localization_threshold'] = value
        base_config['landmarks']['reach_threshold'] = value
        # base_config['landmarks']['random_transitions_percentile'] = value
        # base_config['landmarks']['landmark_mode_interval'] = value
        # base_config['landmarks']['max_landmark_mode_steps'] = value
        # base_config['landmarks']['add_threshold'] = value
        
        mod_config_file = os.path.join(tmp_config_dir, 'run_{}_{}'.format(run_id, os.path.basename(base_file)))  

        with open(mod_config_file, 'w+') as f:
            json.dump(base_config, f)

        proc = subprocess.Popen(['python experiments/landmarks_train.py --config {} --run_ID {} --cuda_idx {} --steps {} --snapshot_gap {}'.format(
            mod_config_file, run_id, cuda_idx, steps, snapshot_gap)], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        time.sleep(sleep)
        run_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base', help='base config file')
    parser.add_argument('--run_id', help='starting run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--snapshot_gap', help='iterations between snapshots', type=int, default=25000)
    parser.add_argument('--steps', help='iterations', type=float, default=1000000)
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--sleep', help='how long to sleep before running another experiment', type=int, default=5)
    args = parser.parse_args()
    run(
        base_file=args.base,
        start_run_id=args.run_id,
        cuda_idx=args.cuda_idx,
        snapshot_gap=args.snapshot_gap,
        steps=args.steps,
        checkpoint=args.checkpoint,
        sleep=args.sleep
    )
