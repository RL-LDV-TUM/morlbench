#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 01, 2016

@author: Johannes Feldmaier <@tum.de>
"""

import logging as log
import numpy as np
import sys

import cPickle as pickle

#log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)


from morl_problems import Deepsea
from morl_agents import QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent, SARSAMorlAgent
from morl_policies import PolicyDeepseaRandom, PolicyDeepseaDeterministic, PolicyDeepseaFromAgent, PolicyDeepseaExpert
from inverse_morl import InverseMORL
from plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_plot2
from dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from experiment_helpers import morl_interact_multiple, morl_interact_multiple_average

import pickle
import time

import matplotlib.pyplot as plt


if __name__ == '__main__':
    problem = Deepsea()

    # scalarization_weights = np.array([0.153, 0.847])
    scalarization_weights = np.array([0.5, 0.5])

    eps = 0.9
    alfa = 0.3
    runs = 500
    interactions = 50000

    agent = QMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSAMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, lmbda=0.9)

    payouts, moves, states = morl_interact_multiple_average(agent, problem, runs=runs, interactions=interactions, max_episode_length=150)
    # payouts, moves, states = morl_interact_multiple(agent, problem, interactions=interactions, max_episode_length=150)

    learned_policy = PolicyDeepseaFromAgent(problem, agent, mode='gibbs')

    filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")
    pickle.dump((payouts, moves, states, problem, agent), open(filename, "wb"))

    ## Plotting ##

    #plt.ion()

    # figure_file_name = 'fig_runs-' + str(interactions) + "-" + agent.name() + ".png"
    policy_plot2(problem, learned_policy)

    #plt.ioff()

    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))



