# Name of the Project: 

MORLBENCH Framework

Multiobjective Reinforcement Learning Benchmark Suite

# Institution:

Chair of data processing @ Technical University of Munich

# Contributors:

Dominik Meyer <meyerd@mytum.de>, 
Johannes Feldmaier <johannes.feldmaier@tum.de>, 
Simon Wölzmüller   <ga35voz@mytum.de>

# Description:

This is the MORLBENCH Framework for algorithm and environment performance 
testing in multiple objective Reinforcement Learning.

written in Python 2.7
 
# Getting Started:

1. After cloning the repository you should run `python setup.py develop` in 
parent directory of the framework. (provided you have python 2.7 on your system
installed. Otherwise download it([Python Website](https://www.python.org/)))

2. You can use every class in `morlbench/morl_agents.py` and 
`morlbench/morl_problems.py` where agents stands for algorithms and problems
for the scenarios you want to test in.

3. For first examples using the framework relate to `morlbench/scratchpad`

# Testing:

To run test type `python -m unittest2 test.py` in `morlbench/test`
