import argparse
import json
import subprocess
import time


def run(base_file, start_run_id, cuda_idx, snapshot_gap, steps, checkpoint, sleep):
    try:
        with open(base_file, 'r') as f:
            base_config = json.load(f)
    except ValueError:
        raise ValueError('Unable to read base config file {}'.format(base_file))

    values = [0.7, 0.8, 0.9, 0.95, 0.99]
    run_id = start_run_id
    for value in values:
        base_config['landmarks']['add_threshold'] = value
        with open(base_file, 'w+') as f:
            json.dump(base_config, f)

        proc = subprocess.Popen(['python experiments/landmarks_train.py --config {} --run_ID {} --cuda_idx {} --steps {} --snapshot_gap {}'.format(
            base_file, run_id, cuda_idx, steps, snapshot_gap)], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        time.sleep(sleep)
        run_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base', help='base config file')
    parser.add_argument('--run_id', help='starting run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--snapshot_gap', help='iterations between snapshots', type=int, default=25000)
    parser.add_argument('--steps', help='iterations', type=float, default=5000000)
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
