#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 11, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_problems import  Deepsea, MountainCarTime, MOPuddleworldProblem, MORLBuridansAssProblem, MORLGridworld, MORLResourceGatheringProblem
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt
import logging as log
import time
if __name__ == '__main__':
    # create Problems
    problem = MORLGridworld()
    problem2 = MORLGridworld()
    # tau is for chebyshev agent
    tau = 1.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, -1.0, -1.0]
    # learning rate
    alf = 0.1
    alfacheb = 0.2
    # Propability of greedy selection
    eps = 0.9
    # create one agent using scalarization method
    chebyagent = MORLScalarizingAgent(problem, [1.0, 0.0, 0.0], alpha=alfacheb, epsilon=eps,
                                      tau=tau, ref_point=ref)
    # create one agent using Hypervolume based Algorithm
    chebyagent2 = MORLScalarizingAgent(problem2, [1.0, 0.0, 0.0], alpha=alfacheb, epsilon=eps,
                                      tau=tau, ref_point=ref)
    # both agents interact (times):
    interactions = 1000
    # make the interactions
    log.info('Playing %i interactions on chebyagent' % interactions)
    payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                            max_episode_length=300)
    # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
    #     str(states[:]) + '\n')
    log.info('Playing %i interactions on hvb agent' % interactions)
    payouts2, moves2, states2 = morl_interact_multiple_episodic(chebyagent2, problem, interactions,
                                                                max_episode_length=299)
    # print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
    #      str(states2[:]) + '\n')

    # extract all volumes of each agent
    agents = [chebyagent2, chebyagent]   # plot the evolution of both agents hypervolume metrics
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

