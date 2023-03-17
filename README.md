# Planning with Sequence Models through Iterative Energy Minimization 

Hongyi Chen\*, Yilun Du\*, Yiye Chen\*, Joshua B. Tenenbaum, Patricio A. Vela

\*equal contribution

A link to our paper can be found on [OpenReview]([https://arxiv.org/abs/2106.01345](https://openreview.net/forum?id=cVFD6qE8gnY)).


Please cite our paper as:

```
@inproceedings{chenplanning,
  title={Planning with Language Models through Iterative Energy Minimization},
  author={Chen, Hongyi and Du, Yilun and Chen, Yiye and Vela, Patricio A and Tenenbaum, Joshua B},
  booktitle={International Conference on Learning Representations}
}
```
## Table of Contents
- [LEAP](#LEAP)
- [Introduction](#Introduction)
- [Install](#install)
- [Usage](#usage)
- [Acknowledgments](#Acknowledgments)


## Introduction
We provide code for evaluate the safety and sample-efficiency of our proposed RL framework.

For safety, we use Safe Set Algorithm (SSA).   
For efficiency, there are more strategies you can choose:  
1, Adapting SSA;  
2, Exploration (PSN, RND, None);  
3, Learning from SSA;  

The safety and efficiency results of all models are shown below
![safety_result](docs/safety_result.png)
![efficiency_result](docs/efficiency_result.png)

We also provide visual tools.
![visulization](docs/visualization.png)


## Install

```
conda create -n safe-rl
conda install python=3.7.9
pip install tensorflow==2.2.1
pip install future
pip install keras
pip install matplotlib
pip install gym
pip install cvxopt
```

## Usage

```
python train.py --display {none, turtle} --explore {none, psn, rnd} --no-qp --no-ssa-buffer
python train.py --display {none, turtle} --explore {none, psn, rnd} --qp --no-ssa-buffer
python train.py --display {none, turtle} --explore {none, psn, rnd} --no-qp --ssa-buffer
```
- `--display` can be either `none` or `turtle` (visulization).
- `--explore` specifies the exploration strategy that the robot uses. 
- `--no-qp` means that we use vanilla SSA.
- `--qp` means that we use adapted SSA.
- `--no-ssa-buffer` means that we use the default learning.
- `--ssa-buffer` means that we use the safe learning from SSA demonstrations.


## Acknowledgments
Part of the simulation environment code is coming from the course CS 7638: Artificial Intelligence for Robotics in GaTech. We get the permission from the lecturor Jay Summet to use this code for research.

