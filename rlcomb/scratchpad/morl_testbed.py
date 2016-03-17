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


from morl_problems import Deepsea, MORLGridworld
from morl_agents import QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent, SARSAMorlAgent, FixedPolicyAgent
from morl_policies import PolicyDeepseaRandom, PolicyDeepseaDeterministic, PolicyFromAgent, PolicyDeepseaExpert
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
    # scalarization_weights = np.array([0.5, 0.5])
    scalarization_weights = np.array([1.0, 0.0])
    # scalarization_weights = np.array([0.0, 1.0])
    # scalarization_weights = np.array([0.9, 0.1])

    eps = 0.6
    alfa = 0.3
    runs = 1
    interactions = 50

    exp_policy = PolicyDeepseaExpert(problem, task='T2')
    # det_policy = PolicyDeepseaDeterministic(problem, policy='P1')
    agent = FixedPolicyAgent(problem, exp_policy)
    # agent = QMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSAMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, lmbda=0.9)

    # payouts, moves, states = morl_interact_multiple_average(agent, problem, runs=runs, interactions=interactions, max_episode_length=150)
    payouts, moves, states = morl_interact_multiple(agent, problem, interactions=interactions, max_episode_length=150)

    learned_policy = PolicyFromAgent(problem, agent, mode='gibbs')
    # learned_policy = PolicyFromAgent(problem, agent, mode='greedy')

    # filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")


    ## Plotting ##

    #plt.ion()

    # figure_file_name = 'fig_runs-' + str(interactions) + "-" + agent.name() + ".png"
    titlestring = agent.name()
    policy_plot2(problem, learned_policy, title=None, filename=titlestring)

    # pickle_file_name = titlestring + '_' + time.strftime("%H%M%S") + '.p'
    # pickle.dump((payouts, moves, states, problem, agent), open(pickle_file_name, "wb"))

    #plt.ioff()

    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))



