#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 03, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

import logging as log
import numpy as np
import sys

import cPickle as pickle

log.basicConfig(level=log.DEBUG)
#log.basicConfig(level=log.INFO)


from rl.morl_problems import Gridworld
from rl.morl_policies import PolicyGridworldExample
from rl.inverse_morl import InverseMORLIRL


if __name__ == '__main__':
    problem = Gridworld(size=10)
    reward_dimension = problem.reward_dimension
    scalarization_weights = np.zeros(reward_dimension)

    policy_optimal = PolicyGridworldExample(problem)

    i_morl = InverseMORLIRL(problem, policy_optimal)
    # scalarization_weights = i_morl.solvep()
    scalarization_weights_alge = i_morl.solvealge()

    # log.info("scalarization weights (with p): %s" % (str(scalarization_weights)))
    # log.info("scalarization weights (without p): %s" % (str(i_morl.solve())))
    # log.info("scalarization weights (without p, sum 1): %s" % (str(i_morl.solve_sum_1())))
    log.info("scalarization weights (alge): %s" % (str(scalarization_weights_alge)))

    tmp = np.dot(problem.R, scalarization_weights_alge)
    # tmp = np.dot(scalarization_weights, problem.R.T)

    sys.exit(0)
