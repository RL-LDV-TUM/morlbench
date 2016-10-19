#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Apr 22, 2016

@author: Dominik Meyer <meyerd@mytum.de>
@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
@author: Simon Woelzmueller   <ga35voz@mytum.de>

    Copyright (C) 2016  Dominik Meyer, Johannes Feldmaier, Simon Woelzmueller

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""



from morlbench.morl_problems import MORLResourceGatheringProblem, MountainCar, MORLGridworld, MORLBuridansAssProblem, Deepsea
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    hypervolume_experiment = False
    comparison_experiment = True
    # create Problem
    problem = MORLGridworld()
    # create an initialize randomly a weight vector
    scalarization_weights = [1.0, 0.0, 0.0]
    # tau is for chebyshev agent
    tau = 0.1
    # ref point is used for Hypervolume calculation
    ref = [-0.1, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.01
    # Propability of epsilon greedy selection
    eps = 0.7
    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau)
    # both agents interact (times):
    interactions = 2000
    n_vectors = 2

    if hypervolume_experiment:
        # make the interactions
        payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=150)
        print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
             str(states[:]) + '\n')

        plot_hypervolume([chebyagent], problem)
    ####################################################################################################################
    # In this experiment we play two agents against each other with different weights and compare hv ###################
    ####################################################################################################################
    if comparison_experiment:
        weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]

        weights = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
                   [0.33, 0.33, 0.33]]
        linagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                        ref_point=ref, tau=tau, function='linear')
        hvb_hypervolumes = []
        cheb_hypervolumes = []
        for i in xrange(len(weights)):
            linagent._w, chebyagent._w = weights[i], weights[i]
            payouts1, moves1, states1 = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                        max_episode_length=150)
            payouts2, moves2, states2 = morl_interact_multiple_episodic(linagent, problem, interactions,
                                                                        max_episode_length=150)
            plot_hypervolume([chebyagent, linagent], problem)
            hvb_hypervolumes.append(max(linagent.max_volumes))
            cheb_hypervolumes.append(max(chebyagent.max_volumes))
            linagent.reset()
            chebyagent.reset()

        fig, ax = plt.subplots()
        width = 0.3
        x = np.arange(1, len(weights)+1)
        ax.bar(x-width, hvb_hypervolumes, width, color='r', label="Linear-Agent")
        ax.bar(x, cheb_hypervolumes, width, color='b', label='Chebyshev-Agent')

        # ax.hist(to_plot, bins=n_vectors, label=['HVBAgent', 'ChebishevAgent'])
        # plt.hist(cheb_hypervolumes, bins=n_vectors,  histtype='bar', alpha=0.5, label='ChebishevAgent')
        plt.axis([0-width, len(weights)+1, 0, 1.1*max([max(cheb_hypervolumes), max(hvb_hypervolumes)])])
        plt.xlabel('weights')
        plt.ylabel('hypervolume maximum')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        weights = [[round(weights[q][i], 2) for i in xrange(len(weights[q]))] for q in xrange(len(weights))]
        ax.set_xticks(x)
        ax.set_xticklabels(weights, rotation=40)
        print weights
        plt.grid(True)
        plt.show()
