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

#log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)


from morl_problems import Deepsea
from morl_agents import QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent
from morl_policies import PolicyDeepseaRandom, PolicyDeepseaDeterministicExample01, PolicyDeepseaFromAgent
from inverse_morl import InverseMORL
from plot_heatmap import policy_plot, transition_map
from dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from experiment_helpers import morl_interact_multiple

import matplotlib.pyplot as plt


if __name__ == '__main__':
    problem = Deepsea()
    reward_dimension = problem.reward_dimension
    scalarization_weights = np.zeros(reward_dimension)

    eps = 0.95
    interactions = 100000

    scalarization_weights_groundtruth = np.array([0.8, 0.2])

    # agent_optimal = PreScalarizedQMorlAgent(problem, scalarization_weights_groundtruth, alpha=0.3, epsilon=eps)
    # payouts, moves, states = morl_interact_multiple(agent_optimal, problem, interactions, max_episode_length=150)

    policy_optimal = PolicyDeepseaDeterministicExample01(problem)
    # policy_optimal = PolicyDeepseaFromAgent(problem=problem, agent=agent_optimal, mode='greedy')
    # policy_plot(problem, policy_optimal)

    i_morl = InverseMORL(problem, policy_optimal)
    scalarization_weights = i_morl.solvep()

    log.info("scalarization weights (with p): %s" % (str(scalarization_weights)))
    log.info("scalarization weights (without p): %s" % (str(i_morl.solve())))
    log.info("scalarization weights (without p, sum 1): %s" % (str(i_morl.solve_sum_1())))
    log.info("scalarization weights (alge): %s" % (str(i_morl.solvealge())))

    sys.exit(0)

    policy = PolicyDeepseaDeterministicExample01(problem)

    # scalarization_weights = np.array([0.153, 0.847])
    # scalarization_weights = np.array([0.2, 0.8])

    # agent = QMorlAgent(problem, scalarization_weights, alpha=0.3, epsilon=eps)
    agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=0.3,
                                    epsilon=eps)
    # agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=0.3, epsilon=eps, lmbda=0.9)

    payouts, moves, states = morl_interact_multiple(agent, problem, interactions, max_episode_length=150)

    plt.ion()

    transition_map(problem, states, moves)
    learned_policy = PolicyDeepseaFromAgent(problem, agent, mode='gibbs')

    plt.ioff()

    # compare agent.policy policy

    figure_file_name = 'fig_runs-' + str(interactions) + "-" + agent.name() + ".png"

    policy_plot(problem, learned_policy, filename=figure_file_name)
