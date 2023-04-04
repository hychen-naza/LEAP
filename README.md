# Planning with Sequence Models through Iterative Energy Minimization 

Hongyi Chen\*, Yilun Du\*, Yiye Chen\*, Joshua B. Tenenbaum, Patricio A. Vela

\*equal contribution

A link to our paper can be found on [OpenReview](https://openreview.net/forum?id=cVFD6qE8gnY).
A link to our project website can be found on [here](https://hychen-naza.github.io/projects/LEAP/index.html).

Please cite our paper as:

```
@inproceedings{chenplanning,
  title={Planning with Sequence Models through Iterative Energy Minimization},
  author={Chen, Hongyi and Du, Yilun and Chen, Yiye and Tenenbaum, Joshua B and Vela, Patricio A},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## Introduction
In this repo, we provide training and evaluation code for Mu<strong>l</strong>tistep <strong>E</strong>nergy-Minimiz<strong>a</strong>tion <strong>P</strong>lanner (LEAP) model in Babyai environment.


## Install

Babyai simulator
```
cd babyai
pip3 install --editable .
```
LEAP training and testing in our environment (should be good in other close version)
```
python=3.7.12
pytorch=1.11.0+cu113
```

## Usage
To reproduce the babyai result
```
bash test_run.sh
```

# Contact

In case of any confusion, please email hchen657@gatech.edu.

# Acknowledgements

The codebase is derived from [decision transformer repo](https://github.com/kzl/decision-transformer).
