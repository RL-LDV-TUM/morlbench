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

# log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)


from rl.morl_problems import Deepsea
from rl.morl_agents import QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent
from rl.morl_policies import PolicyDeepseaRandom, PolicyDeepseaDeterministic, PolicyFromAgent, PolicyDeepseaExpert
from rl.inverse_morl import InverseMORLIRL
from rl.plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_plot2
from rl.dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from rl.experiment_helpers import morl_interact_multiple, morl_interact_multiple_average

import matplotlib.pyplot as plt


if __name__ == '__main__':
    problem = Deepsea(extended_reward=False)
    reward_dimension = problem.reward_dimension
    scalarization_weights_groundtruth = np.zeros(reward_dimension)
    scalarization_weights_groundtruth = np.array([0.4, 0.6])

    eps = 0.4
    alfa = 0.3

    interactions = 50000

    agent_optimal = PreScalarizedQMorlAgent(problem, scalarization_weights_groundtruth, alpha=alfa, epsilon=eps)
    payouts, moves, states = morl_interact_multiple(agent_optimal, problem, interactions, max_episode_length=150)

    # policy_optimal = PolicyDeepseaDeterministic(problem, policy='P5')
    # policy_human = PolicyDeepseaExpert(problem, task='T3')
    # policy_optimal = PolicyDeepseaRandom(problem)
    policy_optimal = PolicyFromAgent(problem=problem, agent=agent_optimal, mode='greedy')
    policy_plot2(problem, policy_optimal)
    # policy_plot2(problem, policy_human)

    i_morl = InverseMORLIRL(problem, policy_optimal)
    # scalarization_weights = i_morl.solvep()
    scalarization_weights_alge = i_morl.solvealge()

    # log.info("scalarization weights (with p): %s" % (str(scalarization_weights)))
    # log.info("scalarization weights (without p): %s" % (str(i_morl.solve())))
    # log.info("scalarization weights (without p, sum 1): %s" % (str(i_morl.solve_sum_1())))
    log.info("scalarization weights (alge): %s" % (str(scalarization_weights_alge)))

    problem2 = Deepsea(extended_reward=False)

    # agent = QMorlAgent(problem, scalarization_weights_alge, alpha=alfa, epsilon=eps)
    agent2 = PreScalarizedQMorlAgent(problem2, scalarization_weights_alge, alpha=alfa, epsilon=eps)
    # agent = SARSALambdaMorlAgent(problem, scalarization_weights_alge, alpha=alfa, epsilon=eps, lmbda=0.9)

    payouts2, moves2, states2 = morl_interact_multiple(agent2, problem2, interactions, max_episode_length=150)

    #plt.ion()

    #transition_map(problem, states, moves)
    learned_policy = PolicyFromAgent(problem2, agent2, mode='gibbs')
    #heatmap_matplot(problem, states)

    policy_plot2(problem2, learned_policy)

    #plt.ioff()

    # compare agent.policy policy

    # figure_file_name = 'fig_runs-' + str(interactions) + "-" + agent.name() + ".png"

    # policy_plot(problem, learned_policy, filename=figure_file_name)
