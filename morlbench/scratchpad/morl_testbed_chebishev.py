from morlbench.morl_problems import MORLResourceGatheringProblem, MORLMountainCar, MORLGridworld, MORLBuridansAssProblem, Deepsea
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
    problem = MORLMountainCar()
    # create an initialize randomly a weight vector
    scalarization_weights = [1.0, 0.0, 0.0]
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.11
    # Propability of epsilon greedy selection
    eps = 0.1
    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau, function='linear')
    # both agents interact (times):
    interactions = 1000
    n_vectors = 10

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
        hvbagent = MORLHVBAgent(problem, alfacheb, eps, ref, weights[0])
        hvb_hypervolumes = []
        cheb_hypervolumes = []
        for i in xrange(n_vectors):
            hvbagent._w, chebyagent._w = weights[i], weights[i]
            payouts1, moves1, states1 = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                        max_episode_length=150)
            payouts2, moves2, states2 = morl_interact_multiple_episodic(hvbagent, problem, interactions,
                                                                        max_episode_length=150)
            hvb_hypervolumes.append(max(hvbagent.max_volumes))
            cheb_hypervolumes.append(max(chebyagent.max_volumes))
            hvbagent.reset()
            chebyagent.reset()

        fig, ax = plt.subplots()
        width = 0.3
        x = np.arange(1, n_vectors+1)
        ax.bar(x-width, hvb_hypervolumes, width, color='r', label="HVB-Agent")
        ax.bar(x, cheb_hypervolumes, width, color='b', label='Chebyshev-Agent')

        # ax.hist(to_plot, bins=n_vectors, label=['HVBAgent', 'ChebishevAgent'])
        # plt.hist(cheb_hypervolumes, bins=n_vectors,  histtype='bar', alpha=0.5, label='ChebishevAgent')
        plt.axis([0-width, n_vectors+1, 0, 1.1*max([max(cheb_hypervolumes), max(hvb_hypervolumes)])])
        plt.xlabel('weights')
        plt.ylabel('hypervolume maximum')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        weights = [[round(weights[q][i], 2) for i in xrange(len(weights[q]))] for q in xrange(len(weights))]
        ax.set_xticks(x)
        ax.set_xticklabels(weights, rotation=40)
        print weights
        plt.grid(True)
        plt.show()
