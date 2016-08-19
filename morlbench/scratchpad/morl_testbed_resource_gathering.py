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


from morlbench.morl_problems import Deepsea, MORLGridworld, MOPuddleworldProblem, MORLResourceGatheringProblem
from morlbench.morl_agents import MORLHLearningAgent, MORLRLearningAgent, MORLHVBAgent, MORLScalarizingAgent
from morlbench.morl_policies import PolicyDeepseaRandom, PolicyDeepseaDeterministic, PolicyFromAgent, PolicyDeepseaExpert
from morlbench.inverse_morl import InverseMORL
from morlbench.plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_plot2, policy_heat_plot
from morlbench.dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic
from morlbench.plotting_stuff import plot_hypervolume
import pickle
import time

import matplotlib.pyplot as plt


if __name__ == '__main__':
    problem = MORLResourceGatheringProblem()

    # scalarization_weights = np.array([0.153, 0.847])
    # scalarization_weights = np.array([0.5, 0.5])
    scalarization_weights = np.array([0.5, 0.5, 0.0])
    # scalarization_weights = np.array([0.0, 1.0])
    # scalarization_weights = np.array([0.9, 0.1])

    eps = 0.8
    alfa = 0.4
    runs = 2
    interactions = 100


    agent = MORLScalarizingAgent(problem, scalarization_weights, alfa, eps, 4.0, [-1.0, -1.0, -1.0])



    # payouts, moves, states = morl_interact_multiple_average_episodic(agent, problem, runs=runs, interactions=interactions, max_episode_length=150)
    payouts, moves, states = morl_interact_multiple_episodic(agent, problem, interactions=interactions, max_episode_length=150)

    learned_policy = PolicyFromAgent(problem, agent, mode='gibbs')
    # learned_policy = PolicyFromAgent(problem, agent, mode='greedy')

    # filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")


    ## Plotting ##

    # plt.ion()

    # figure_file_name = 'fig_runs-' + str(interactions) + "-" + agent.name() + ".png"
    titlestring = agent.name()
    # policy_plot2(problem, learned_policy, title=None, filename=titlestring)
    # policy_heat_plot(problem, learned_policy, states)
    # pickle_file_name = titlestring + '_' + time.strftime("%H%M%S") + '.p'
    # pickle.dump((payouts, moves, states, problem, agent), open(pickle_file_name, "wb"))

    # plt.ioff()
    plot_hypervolume([agent], problem)
    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))



