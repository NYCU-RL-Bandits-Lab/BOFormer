# [ICLR 2025] BOFormer: Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL

[Paper](https://openreview.net/forum?id=UnCKU8pZVe&noteId=UnCKU8pZVe) | [Project Page](https://hungyuheng.github.io/BOFormer/)

## Abstract
Bayesian optimization (BO) offers an efficient pipeline for optimizing black-box functions with the help of a Gaussian process prior and an acquisition function (AF). Recently, in the context of single-objective BO, learning-based AFs witnessed promising empirical results given its favorable non-myopic nature. Despite this, the direct extension of these approaches to multi-objective Bayesian optimization (MOBO) suffer from the hypervolume identifiability issue, which results from the non-Markovian nature of MOBO problems. To tackle this, inspired by the non-Markovian RL literature and the success of Transformers in language modeling, we present a generalized deep Q-learning framework and propose BOFormer, which substantiates this framework for MOBO via sequence modeling. Through extensive evaluation, we demonstrate that BOFormer constantly achieves better performance than the benchmark rule-based and learning-based algorithms in various synthetic MOBO and real-world multi-objective hyperparameter optimization problems.



### Environment Installation
```
Run install.sh to set up the Conda environment.

#### Installing OptFormer
pip install sentencepiece

#### Installing qTransformer
pip install transformers==4.12.1
```

### Usage
```
Refer to run.sh for additional scripts and usage details.
```
