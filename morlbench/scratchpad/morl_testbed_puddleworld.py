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


from morlbench.morl_problems import MOPuddleworldProblem
from morlbench.morl_agents import MORLScalarizingAgent, QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent,\
    SARSAMorlAgent, MORLHVBAgent
from morlbench.morl_policies import PolicyFromAgent, PolicyGridworld
from morlbench.inverse_morl import InverseMORLIRL
from morlbench.plot_heatmap import policy_plot2, transition_map, heatmap_matplot, policy_heat_plot
from morlbench.dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic
from morlbench.plotting_stuff import plot_hypervolume

import pickle
import time

import matplotlib.pyplot as plt


if __name__ == '__main__':

    runs = 1

    saved_weights = []
    plt.ion()

    for i in xrange(runs):

        problem = MOPuddleworldProblem()
        scalarization_weights = np.array([1.0, 0.0])

        eps = 0.2
        alfa = 0.1
        tau = 2.0

        interactions = 1000

        # agent = QMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
        # agent = PreScalarizedQMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
        # agent = SARSAMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps)
        # agent = SARSALambdaMorlAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, lmbda=0.9)
        agent = MORLScalarizingAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, tau=tau, lmbda=0.9,
                                  ref_point=[-1.0, -1.0])
        # agent = MORLHVBAgent(problem, alpha=alfa, epsilon=0.9, ref=[-1.0, -1.0], scal_weights=[1.0, 10.0])


        # payouts, moves, states = morl_interact_multiple_average_episodic(agent, problem, runs=runs, interactions=interactions, max_episode_length=150)
        payouts, moves, states = morl_interact_multiple_episodic(agent, problem, interactions=interactions, max_episode_length=300)

        log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))

        # show_exploration(states, problem.n_states)
        # learned_policy = PolicyFromAgent(problem, agent, mode='gibbs')
        # learned_policy = PolicyFromAgent(problem, agent, mode='None')
        learned_policy = PolicyFromAgent(problem, agent, mode='greedy')
        # learned_policy = PolicyGridworld(problem, policy='DIAGONAL')
        # learned_policy = PolicyGridworld(problem, policy='RIGHT')
        # learned_policy = PolicyGridworld(problem, policy='DOWN')

        # filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")

        # pickle.dump((payouts, moves, states, problem, agent), open('test_pickle.p', "wb"))

        ## Plotting ##
        expName = "Exp1-"
        for w in range(len(scalarization_weights)):
            expName += str(scalarization_weights[w])+"-"
        expName += "_run_"
        fName1 = expName + str(i) + '_learned'
        fName2 = expName + str(i) + '_retrieved'

        policy_plot2(problem, learned_policy)
        policy_heat_plot(problem, learned_policy, states, filename=fName1)
        plot_hypervolume([agent], problem)


