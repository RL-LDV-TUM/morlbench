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
from morl_problems import MountainCar
from morl_agents import SARSAMorlAgent
from morl_agents import QMorlAgent
from morl_agents import DeterministicAgent
from morl_agents import NFQAgent
from experiment_helpers import morl_interact_multiple
from plot_heatmap import transition_map,heatmap_matplot


if __name__ == '__main__':
    problem = Deepsea(gamma=0.9)
    # problem = MountainCar()
    reward_dimension = problem.reward_dimension
    scalarization_weights = np.zeros(reward_dimension)
    scalarization_weights[0] = 0.5
    scalarization_weights[1] = 0.5
    agent = SARSAMorlAgent(problem, scalarization_weights=scalarization_weights,
                           alpha=0.1, epsilon=0.7)
    # agent = QMorlAgent(problem, scalarization_weights=scalarization_weights,
    #                        alpha=0.1, epsilon=0.6)
    # agent = DeterministicAgent(problem)
    # agent = NFQAgent(problem, scalarization_weights, epsilon=0.8)

    interactions = 5000

    log.info('Playing ...')
    log.info('%s' % (str(agent)))
    log.info('%s' % (str(problem)))

    #_, payouts = morl_interact_multiple(agent, problem, interactions)
    payouts, moves, states = morl_interact_multiple(agent, problem, interactions, max_episode_length=150)

    # Save payouts, moves, states to pickle file
    pickle.dump((payouts, moves, states), open("results.p", "wb"))

    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))

    # Plot transition map and pause program during display
    transition_map(problem=problem, states=states, moves=moves)
    heatmap_matplot(problem, states)


