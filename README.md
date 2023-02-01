# Local Causal Discovery for Estimating Causal Effects


### Overview

This repository contains the implementation for the paper: <br>
__Local Causal Discovery for Estimating Causal Effects__ <br>
_Conference on Causal Learning and Reasoning (CLeaR), 2023_ <br>
Shantanu Gupta, David Childers, Zachary Lipton

### Code and Datasets

The `causal_discovery` folder contains the code for the various causal discovery
algorithms used in our work:
 - `pc_alg.py`: An implementation of the PC algorithm 
    (code adapted from the [pcalg](https://github.com/keiichishima/pcalg/blob/master/pcalg.py) library).
 - `sd_alg.py`: An implementation of the SD algorithm which recursively runs the PC algorithm
    locally starting from the treatment, its neighbors, and so on.
 - `mb_by_mb.py`: An implementation of the MB-by-MB algorithm ([Wang et. al., 2014](https://www.sciencedirect.com/science/article/abs/pii/S0167947314000802)) which 
 performs local causal discovery by sequentially finding Markov blankets and local
 structures within the Markov blankets.
 - `ldecc.py`: An implementation of the LDECC algorithm proposed in our work.

The following Jupyter notebooks contain example usages of the various algorithms
and code for running the experiments in our paper:
 - `Results_on_synthetic_linear_graphs.ipynb`: Code for experiments on synthetic linear
    Gaussian graphs (Fig. 10).
 - `Results_on_semi_synthetic_linear_Gaussian_graphs.ipynb`: Code for experiments on semi-synthetic
   linear Gaussian graphs from [*bnlearn*](https://www.bnlearn.com/) (Figs. 11, 17).
 - `Results_on_synthetic_linear_Erdos_Renyi_graphs.ipynb`: Code for experiments on synthetic
   Erdos-Renyi linear Gaussian graphs (Fig. 16).
 - `Results_on_semi_synthetic_discrete_graphs.ipynb`: Code for experiments on three semi-synthetic
    discrete graphs from [*bnlearn*](https://www.bnlearn.com/) (Fig. 18).

The `data` folder contains the [*bnlearn*](https://www.bnlearn.com/) graphs
used in our experiments.


### Citation
If you find this work useful, please consider citing our work:
```bib
@inproceedings{gupta2023local,
  title={Local Causal Discovery for Estimating Causal Effects},
  author={Gupta, Shantanu and Childers, David and Lipton, Zachary C.},
  booktitle={Conference on Causal Learning and Reasoning},
  year={2023}
}
```
