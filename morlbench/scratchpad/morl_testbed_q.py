#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 11, 2016

@author: Johannes Feldmaier <@tum.de>

Testbed for basic q-learning in the deep sea
and gridworld environment.

"""

import logging as log
import numpy as np

import time
import sys
import cPickle as pickle

# log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)

from morlbench.morl_problems import Deepsea, MORLResourceGatheringProblem, MORLGridworld, MORLGridworldTime, MORLBuridansAssProblem, MOPuddleworldProblem, MountainCarTime
from morlbench.morl_agents import QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent, SARSAMorlAgent,\
    MORLScalarizingAgent, MORLHVBAgent
from morlbench.morl_policies import PolicyDeepseaRandom, PolicyDeepseaDeterministic, PolicyFromAgent, PolicyDeepseaExpert
from morlbench.inverse_morl import InverseMORL
from morlbench.plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_plot2, policy_heat_plot
from morlbench.dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic
from morlbench.plotting_stuff import plot_hypervolume


if __name__ == '__main__':
    problem = MORLBuridansAssProblem()

    # Define scalarization weights
    scalarization_weights = np.array([1.0, 0.0, 0.0]) # go to reward 50
    # scalarization_weights = np.array([0.13, 0.87]) # go to reward 16
    # scalarization_weights = np.array([0.1, 0.9]) # go to reward 16
    # scalarization_weights = np.array([0.5, 0.5])
    # scalarization_weights = np.array([1.0, 0.0])
    # scalarization_weights = np.array([0.0, 1.0])
    # scalarization_weights = np.array([0.9, 0.1])
    # scalarization_weights = np.array([1.0, 0.0, 0.0])

    # define params
    eps = 0.1
    alfa = 0.1
    runs = 3
    interactions = 500
    episode_length = 150
    tau = 4.0  # only for Chebyshev and deepsea
    gamma = 0.9
    ref = [-1.0, -1.0, -1.0]  # reference point for hypervolume calculation
    # Select a learning agent:
    # agent1 = QMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSAMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
    # agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, lmbda=0.9)
    agent = MORLScalarizingAgent(problem, scalarization_weights, alfa, eps, tau, ref_point=ref, gamma=gamma, function='chebishev')
    # agent = MORLHVBAgent(problem, alfa, eps, ref, scalarization_weights)
    # Run the experiment one time for the given number of interactions
    payouts, moves, states = morl_interact_multiple_episodic(agent, problem, interactions=interactions,
                                                             max_episode_length=episode_length)

    # only for multidimensional state problems (resource gathering, buridans ass)
    states = problem.create_plottable_states(states)
    # Repeat experiment for "runs" times and average the results
    # payouts, moves, states = morl_interact_multiple_average_episodic(agent, problem, runs=runs, interactions=interactions,
    #                                                          max_episode_length=episode_length)

    # Get learned policy from agent using the defined method
    # learned_policy = PolicyFromAgent(problem, agent, mode='gibbs')
    learned_policy = PolicyFromAgent(problem, agent, mode='greedy')

    # filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")

    ## Plotting ##

    #plt.ion()

    # figure_file_name = 'fig_runs-' + str(interactions) + "-" + agent.name() + ".png"
    titlestring = agent.name()

    policy_plot2(problem, learned_policy, title=None, filename=titlestring)
    policy_heat_plot(problem, learned_policy, states, filename=titlestring)
    # agents = [agent1, agent2, agent3]
    plot_hypervolume([agent], problem)
    print learned_policy._pi
    # pickle_file_name = titlestring + '_' + time.strftime("%H%M%S") + '.p'
    # pickle.dump((payouts, moves, states, problem, agent), open(pickle_file_name, "wb"))

    #plt.ioff()

    #log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))



