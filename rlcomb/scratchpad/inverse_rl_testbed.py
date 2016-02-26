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

    policy = PolicyDeepseaDeterministicExample01(problem)

    # i_morl = InverseMORL(problem, policy)
    # scalarization_weights = i_morl.solve()
    # scalarization_weights = np.array([0.153, 0.847])
    scalarization_weights = np.array([0.2, 0.8])

    # agent = QMorlAgent(problem, scalarization_weights, alpha=0.5, epsilon=0.7)
    # agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=0.3, epsilon=0.95)
    agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=0.3, epsilon=0.3, lmbda=0.9)
    interactions = 5000
    payouts, moves, states = morl_interact_multiple(agent, problem, interactions, max_episode_length=150)

    plt.ion()

    transition_map(problem, states, moves)
    learned_policy = PolicyDeepseaFromAgent(problem, agent)

    plt.ioff()

    # compare agent.policy policy
    policy_plot(problem, learned_policy)
