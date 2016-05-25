#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mai 25 2015

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_problems import MOPuddleworldProblem, MORLBurdiansAssProblem, MORLGridworld, Deepsea
from morlbench.morl_agents import MORLChebyshevAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple

import numpy as np
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':

    problem = MORLGridworld()
    # create an initialize randomly a weight vector
    scalarization_weights = np.zeros(problem.reward_dimension)
    scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)], len(scalarization_weights))
    # tau is for chebyshev agent
    tau = 0.1
    # ref point is used for Hypervolume calculation
    ref = [-0.1, ]*problem.reward_dimension
    # learning rate
    alf = 0.1
    alfacheb = 0.4
    alfahvb = 0.1

    # Propability of epsilon greedy selection
    eps = 0.1
    # both agents interact (times):
    interactions = 200
    agents = []
    # list of volumes
    vollist = []
    # 6 agents with each different weights
    agents.append(MORLChebyshevAgent(problem, [1.0, 0.0, 0.0], alpha=alf, epsilon=eps,
                                     tau=tau, ref_point=ref))
    agents.append(MORLChebyshevAgent(problem, [0.0, 1.0, 0.0], alpha=alf, epsilon=eps,
                                     tau=tau, ref_point=ref))
    agents.append(MORLChebyshevAgent(problem, [0.5, 0.5, 0.0], alpha=alf, epsilon=eps,
                                     tau=tau, ref_point=ref))
    agents.append(MORLChebyshevAgent(problem, [0.0, 0.0, 1.0], alpha=alf, epsilon=eps,
                                     tau=tau, ref_point=ref))
    agents.append(MORLChebyshevAgent(problem, [0.0, 0.5, 0.5], alpha=alf, epsilon=eps,
                                     tau=tau, ref_point=ref))
    agents.append(MORLChebyshevAgent(problem, [0.5, 0.0, 0.5], alpha=alf, epsilon=eps,
                                     tau=tau, ref_point=ref))
    agents.append(MORLChebyshevAgent(problem, [0.33, 0.33, 0.33], alpha=alf,
                                     epsilon=eps, tau=tau, ref_point=ref))

    # interact with each
    for agent in agents:
        p, a, s = morl_interact_multiple(agent, problem, interactions,
                                         max_episode_length=150)
        # store all volumes containing (0,0)
        maxvol = [0]
        maxvol.extend(agent.max_volumes)
        vollist.append(maxvol)

    # cut longer lists
    length = min([len(x) for x in vollist])
    for lists in vollist:
        del lists[length:]
    # create x vectors
    x = np.arange(length)
    # colour vector
    colours = ['r', 'b', 'g', 'k', 'y', 'm', 'c']
    for u in range(len(vollist)):
        # printed name for label
        weights = agents[u]._w
        name = 'weights:'
        for i in range(len(weights)):
            name += str(weights[i])+'_'
        # no last underline
        name = name[:len(name)-1]
        # plotting
        plt.plot(x, vollist[u], colours[u], label=name)
    # size of axes
    plt.axis([0-0.01*len(x), len(x), 0, 1.1*max([max(x) for x in vollist])])
    # position the legend
    plt.legend(loc='lower right', frameon=False)
    # show!
    plt.show()