# Successor Feature Landmarks for Long-Horizon Goal-Conditioned Reinforcement Learning

This is the codebase for the Successor Feature Landmarks project. The associated paper which has been accepted to NeurIPS 2021 can be found at <https://arxiv.org/pdf/2111.09858.pdf>.

## Running the code

### Installation
1. Clone this repository to the local machine.

2. Install the anaconda environment that is compatible with your machine.

```
conda env create -f linux_[cpu|cuda9|cuda10_1|cuda10_1].yml
source activate rlpyt
```

3. Install rlpyt as an editable python package

```
pip install -e .
```

4. Install additional packages (some are related to your desired environment such as gym). An example `requirements.txt` file is included. 

```
pip install -r requirements.txt
```

Details on environments and how to install them
- MiniGrid: 
    - <https://github.com/maximecb/gym-minigrid>
- ViZDoom:
    - <https://github.com/mwydmuch/ViZDoom>
    - <https://github.com/nsavinov/gym-vizdoom>

### Executing experiments

#### MiniGrid

Example run command for MiniGrid's MultiRoom environment

```
git checkout update-minigrid

python experiments/landmarks_train.py --config experiments/minigrid-configs/multiroom/base-4rooms.json --run_ID 0 --cuda_idx 0 --steps 500000 --gpu_fraction 0.3
```

#### ViZDoom

Example run command for SPTM's Train environment

```
git checkout eval

python experiments/vizdoom_eval_original.py --config experiments/configs/memory-train-full-update-goals.json --run_ID 1 --cuda_idx 1 --steps 2000000 --gpu_fraction 0.5
```

### Changing configurations

You can find the experiments configurations in this directory: `experiments/configs`. They contain hyperparameters, paths to pretrained model weights (we use SPTM's network as a feature extractor in ViZDoom), paths to generated (start, goal) pairs for evaluation, and other miscellaneous parameters.

<https://github.com/2016choang/sfl/blob/eval/playground/vizdoom-eval.ipynb> contains example code for generating new (start, goal) pairs for evaluation.

## Cite our paper

Please consider citing our paper if you end up using our work.

```
@inproceedings{Hoang:NIPS2021:SFL,
    author = {Hoang, Christopher and Sohn, Sungryull and Choi, Jongwook and Carvalho, Wilka and Lee, Honglak},
    title = {{Successor Feature Landmarks for Long-Horizon Goal-Conditioned Reinforcement Learning}},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year = {2021}
}
```

## Acknowledgements

- codebase built upon BAIR's [rlpyt](https://github.com/astooke/rlpyt)
- some code and pretrained models from [SPTM](https://github.com/nsavinov/SPTM)
