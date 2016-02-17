#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
"""

import logging as log
import numpy as np

import cPickle as pickle

#log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)

from morl_problems import Deepsea
from morl_agents import SARSAMorlAgent
from morl_agents import DeterministicAgent
from experiment_helpers import morl_interact_multiple
from plot_heatmap import transition_map,heatmap_matplot


if __name__ == '__main__':
    problem = Deepsea()
    reward_dimension = problem.reward_dimension
    scalarization_weights = np.zeros(reward_dimension)
    scalarization_weights[0] = 0.953
    scalarization_weights[1] = 0.047
    # agent = SARSAMorlAgent(problem, scalarization_weights=scalarization_weights,
    #                        alpha=0.1, gamma=0.9, epsilon=0.9)
    agent = DeterministicAgent(problem)

    interactions = 10

    log.info('Playing ...')
    log.info('%s' % (str(agent)))
    log.info('%s' % (str(problem)))

    #_, payouts = morl_interact_multiple(agent, problem, interactions)
    payouts, moves, states = morl_interact_multiple(agent, problem, interactions, trials=1000)

    pickle.dump((payouts, moves, states), open("results.p", "wb"))

    transition_map(problem=problem, states=states, moves=moves)
    #heatmap_matplot(problem, states)

    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))
