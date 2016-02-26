#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 25, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

import logging as log
import numpy as np
import sys

import cPickle as pickle

log.basicConfig(level=log.DEBUG)
#log.basicConfig(level=log.INFO)

from morl_problems import Deepsea
from morl_agents import QMorlAgent
from morl_policies import PolicyDeepseaRandom
from dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from experiment_helpers import morl_interact_multiple


if __name__ == '__main__':
    problem = Deepsea()
    reward_dimension = problem.reward_dimension
    scalarization_weights = np.zeros(reward_dimension)

    policy = PolicyDeepseaRandom(problem)

    scalarization_weights = InverseMORL(problem, policy)

    agent = QMorlAgent(problem, scalarization_weights)

    # compare agent.policy policy
