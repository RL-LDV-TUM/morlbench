#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 03, 2016

@author: Dominik Meyer <meyerd@mytum.de>


    Copyright (C) 2016  Dominik Meyer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import logging as log
import numpy as np
import sys

import cPickle as pickle

log.basicConfig(level=log.DEBUG)
#log.basicConfig(level=log.INFO)


from morlbench.morl_problems import Gridworld
from morlbench.morl_policies import PolicyGridworldExample
from morlbench.inverse_morl import InverseMORLIRL


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
