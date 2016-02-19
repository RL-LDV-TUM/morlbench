#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 19, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

import logging as log
import numpy as np

import cPickle as pickle

#log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)

from morl_problems import Deepsea
from morl_agents import TDMorlAgent
from morl_policies import PolicyDeepseaRandom
from dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from experiment_helpers import morl_interact_multiple


if __name__ == '__main__':
    problem = Deepsea()
    reward_dimension = problem.reward_dimension
    scalarization_weights = np.zeros(reward_dimension)
    scalarization_weights[0] = 0.5
    scalarization_weights[1] = 0.5

    policy = PolicyDeepseaRandom(problem)
    agent = TDMorlAgent(problem, scalarization_weights, policy, alpha=0.1)

    solver_dynamic_inverse = MORLDynamicProgrammingInverse(problem, policy)
    solver_dynamic_pe = MORLDynamicProgrammingPolicyEvaluation(problem, policy)

    inverseV = solver_dynamic_inverse.solve()
    peV = solver_dynamic_pe.solve(max_iterations=100000)

    scalarized_inverseV = np.dot(inverseV, scalarization_weights)
    scalarized_peV = np.dot(peV, scalarization_weights)

    interactions = 10000
    trials = 1000

    log.info('Playing ...')
    log.info('%s' % (str(agent)))
    log.info('%s' % (str(problem)))

    payouts, moves, states = morl_interact_multiple(agent, problem, interactions, max_episode_length=trials)

    tdV = agent._V

    log.info('||scalarized_peV - scalarized_inverseV|| = %f' % (np.linalg.norm(scalarized_peV - scalarized_inverseV)))
    log.info('||tdV - scalarized_peV|| = %f' % (np.linalg.norm(tdV - scalarized_peV)))
    log.info('||tdV - scalarized_inverseV|| = %f' % (np.linalg.norm(tdV - scalarized_inverseV)))

    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))
