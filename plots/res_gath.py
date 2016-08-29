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
    scalarization_weights = np.array([1.0, 1.0 , 0.0])
    # scalarization_weights = np.array([0.0, 1.0])
    # scalarization_weights = np.array([0.9, 0.1])

    eps = 0.9
    alfa = 0.1
    runs = 1
    interactions = 100
    max_steps = 200
    tau = 1.0
    ref_point = [-1.0, ]*problem.reward_dimension
    hvbagent = MORLHVBAgent(problem, alpha=alfa, epsilon=0.1, ref=ref_point, scal_weights=[1.0, 10.0])

    payouts, moves, states = morl_interact_multiple_episodic(hvbagent, problem, interactions=interactions)
    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))

    learned_policy = PolicyFromAgent(problem, hvbagent, mode='greedy')

    states = problem.create_plottable_states(states)

    policy_plot2(problem, learned_policy, title=None, filename=None)
    policy_heat_plot(problem, learned_policy, states)
    plot_hypervolume([hvbagent], problem)
    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))



