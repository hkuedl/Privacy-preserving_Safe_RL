# Privacy-preserving Safe Reinforcement Learning for the Dispatch of a Local Energy Community

_Propose a privacy-preserving safe soft actor-critic (PSSAC) framework for the coordinated dispatch of an LEC, which incorporates encryption modules and safety modules._

Code for paper "Privacy-Preserving Safe Reinforcement Learning for the Dispatch of a Local Energy Community".

Authors: Haoyuan Deng, Ershun Du, and Yi Wang.


## Prerequisites
- Python 
- Conda


### Initial packages include
  - python = 3.8.18
  - numpy
  - pandas
  - scikit-learn
  - gurobipy
  - matplotlib
  - pytorch
  - math
  - tqdm
  - copy
  - time


## Experiments 

We have provided corresponding codes for all policy training and testing in the paper. There are two kinds of programs that respond to small-scale networks and large networks:
  - Small-scale network policy training: main_Operators_AC.py
  - Small-scale network policy testing: Test_AC.ipynb
  - Large-scale network policy training: main_Operators_AC_LN.py
  - Large-scale network policy testing: Test_AC_LN.ipynb














