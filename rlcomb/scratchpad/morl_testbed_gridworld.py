#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 03, 2016

@author: Johannes Feldmaier <@tum.de>
"""

import logging as log
import numpy as np
import sys

import cPickle as pickle

#log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)


from morl_problems import Deepsea, MORLGridworld, MORLGridworldTime
from morl_agents import QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent, SARSAMorlAgent
from morl_policies import PolicyFromAgent, PolicyGridworld
from inverse_morl import InverseMORLIRL
from plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_heat_plot
from dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from experiment_helpers import morl_interact_multiple, morl_interact_multiple_average

import pickle
import time

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # problem = MORLGridworld()
    problem = MORLGridworldTime()

    scalarization_weights = np.array([10.0, 0.0, 0.0, 1.0])

    eps = 0.4
    alfa = 0.3
    runs = 1
    interactions = 10000

    # agent = QMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSAMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, lmbda=0.9)
    #
    # payouts, moves, states = morl_interact_multiple_average(agent, problem, runs=runs, interactions=interactions, max_episode_length=150)
    payouts, moves, states = morl_interact_multiple(agent, problem, interactions=interactions, max_episode_length=150)
    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))

    # learned_policy = PolicyFromAgent(problem, agent, mode='gibbs')
    learned_policy = PolicyFromAgent(problem, agent, mode=None)
    # learned_policy = PolicyFromAgent(problem, agent, mode='greedy')
    # learned_policy = PolicyGridworld(problem, policy='DIAGONAL')
    # learned_policy = PolicyGridworld(problem, policy='RIGHT')
    # learned_policy = PolicyGridworld(problem, policy='DOWN')

    # filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")

    # pickle.dump((payouts, moves, states, problem, agent), open('test_pickle.p', "wb"))

    i_morl = InverseMORLIRL(problem, learned_policy)
    scalarization_weights_alge = i_morl.solvealge()
    #
    log.info("scalarization weights (alge): %s" % (str(scalarization_weights_alge)))
    #
    #
    problem2 = MORLGridworldTime()
    agent2 = PreScalarizedQMorlAgent(problem2, scalarization_weights_alge, alpha=alfa, epsilon=eps)
    payouts2, moves2, states2 = morl_interact_multiple(agent2, problem2, interactions=interactions, max_episode_length=150)
    log.info('Average Payout: %s' % (str(payouts2.mean(axis=0))))

    # learned_policy2 = PolicyFromAgent(problem2, agent2, mode='gibbs')
    learned_policy2 = PolicyFromAgent(problem2, agent2, mode=None)

    ## Plotting ##

    plt.ion()
    # policy_plot2(problem, learned_policy)
    policy_heat_plot(problem, learned_policy, states)
    plt.ioff()
    # policy_plot2(problem2, learned_policy2)
    policy_heat_plot(problem2, learned_policy2, states2)


