#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 11, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_problems import  Deepsea, MOPuddleworldProblem, MORLBuridansAssProblem, MORLGridworld, MORLResourceGatheringProblem
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt
import logging as log
"""
This experiment shows the use of the hypervolume metric as quality indicator
There are two parts of the experiment:
    - It takes one Problem and lets train two different agents (1) upon it.
    - It takes one Problem and lets train six equal agents with different weights (2) on it
At the end it shows the evolution in two seperate plots
You can:
    - change reference point
    - play with the chebishev learning parameter tau
    - change weights of chebishev agent
    - play with learning rates
    - alternate epsilon
    - train more or less often by adjusting interactions
    and see what happens to the learning process
Attention:
    - High epsilons may slow down learning process
    - Too small learning rates cause little impact and small learning effect
    - Too big learning rates cause too big impact on learning process
    - Sum of weight vector elements should equal 1
    - learning rate alfa, epsilon and lambda are parameters out of [0, 1]
"""
if __name__ == '__main__':
    # create Problem
    experiment_1 = True
    experiment_2 = False
    problem = MORLBuridansAssProblem()
    # create an initialize randomly a weight vector
    scalarization_weights = np.zeros(problem.reward_dimension)
    scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)], len(scalarization_weights))
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, -1.0, -1.0]
    # learning rate
    alf = 0.2
    alfacheb = 0.2
    alfahvb = 0.01
    n_vectors = 5

    # Propability of epsilon greedy selection
    eps = 0.1
    # create one agent using scalarization method
    chebyagent = MORLScalarizingAgent(problem, [1.0, 0.0, 0.0], alpha=alfacheb, epsilon=eps,
                                      tau=tau, ref_point=ref, function='linear')
    # create one agent using Hypervolume based Algorithm
    hvbagent = MORLHVBAgent(problem, alpha=alfahvb, epsilon=0.1, ref=ref, scal_weights=[1.0, 10.0])
    # both agents interact (times):
    interactions = 100
    if experiment_1:
        # make the interactions
        log.info('Playing %i interactions on chebyagent' % interactions)
        payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=150)
        # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
        #     str(states[:]) + '\n')
        log.info('Playing %i interactions on hvb agent' % interactions)
        payouts2, moves2, states2 = morl_interact_multiple_episodic(hvbagent, problem, interactions,
                                                                    max_episode_length=150)
        # print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
        #      str(states2[:]) + '\n')

        # extract all volumes of each agent
        agents = [hvbagent, chebyagent]   # plot the evolution of both agents hypervolume metrics
        plot_hypervolume(agents, problem, name='agent')
        plt.figure()
        length = min([len(payouts), len(payouts2)])
        x = np.arange(length)
        if length != len(payouts):
            payouts = payouts[:length]
        else:
            payouts2 = payouts2[:length]
        plt.plot(x, payouts, 'r', label='cheb')
        plt.plot(x, payouts2, 'b', label='hvb')
        plt.show()

    if experiment_2:
        # list of agents with different weights
        agent_group = []
        # list of volumes
        vollist = []
        # 6 agents with each different weights
        # weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5],
        #           [0.5, 0.0, 0.5], [0.33, 0.33, 0.33]]
        weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]
        for weight in weights:
            agent_group.append(MORLScalarizingAgent(problem, weight, alpha=alfacheb, epsilon=eps,
                                                    tau=tau, ref_point=ref))

        # interact with each
        log.info('Playing %i interactions on %i chebyagents' % (interactions, len(agent_group)))
        for agent in agent_group:
            p, a, s = morl_interact_multiple_episodic(agent, problem, interactions,
                                                      max_episode_length=150)
        # plot the evolution of hv of every weights agent
        plot_hypervolume(agent_group, agent_group[0]._morl_problem, name='weights')

