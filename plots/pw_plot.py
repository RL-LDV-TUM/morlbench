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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


if __name__ == '__main__':

    runs = 1

    saved_weights = []
    plt.ion()

    for i in xrange(runs):

        problem = MOPuddleworldProblem(size=20)
        scalarization_weights = np.array([1.0, 0.0])
        max_episode_l = 200

        alfa = 0.1
        tau = 1.0

        interactions = 50

        eps=0.9

        agent = MORLScalarizingAgent(problem, scalarization_weights, alpha=alfa, epsilon=eps, tau=tau, lmbda=1.0,
                                  ref_point=[-1.0, -1.0])

        payouts, moves, states = morl_interact_multiple_episodic(agent, problem, interactions=interactions, max_episode_length=max_episode_l)
        agent.create_scalar_Q_table()
        x = [w for w in xrange(problem._size)]
        y = [d for d in xrange(problem._size)]
        x, y = np.meshgrid(x, y)
        z = np.array([max([agent.Qs[s, a] for a in xrange(problem.n_actions)]) for s in xrange(problem.n_states)])
        z = z.reshape(problem._size, problem._size)

        fig, ax = plt.subplots()
        ax.imshow(z, interpolation='nearest')
        # plt.colorbar()
        plt.grid()
        plt.show()
        log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))


        learned_policy = PolicyFromAgent(problem, agent, mode='greedy')
        ## Plotting ##
        expName = "Exp1-"
        for w in range(len(scalarization_weights)):
            expName += str(scalarization_weights[w])+"-"
        expName += "_run_"
        fName1 = expName + str(i) + '_learned'
        fName2 = expName + str(i) + '_retrieved'

        policy_plot2(problem, learned_policy)

        plot_hypervolume([agent], problem)


