from morlbench.morl_problems import Deepsea
from morlbench.morl_agents import MORLChebyshevAgent
from morlbench.experiment_helpers import morl_interact_multiple
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create Problem
    problem = Deepsea()
    # create an initialize randomly a weight vector
    scalarization_weights = [0.0, 1.0]
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-25.0, -1.0]
    # learning rate
    alfacheb = 0.1
    # Propability of epsilon greedy selection
    eps = 0.6
    # create one agent using chebyshev scalarization method
    chebyagent = MORLChebyshevAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                    ref_point=ref, tau=tau,)
    # both agents interact (times):
    interactions = 1000
    # make the interactions
    payouts, moves, states = morl_interact_multiple(chebyagent, problem, interactions,
                                                    max_episode_length=150)
    # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
    #     str(states[:]) + '\n')

    # extract all volumes of each agent
    a_list = chebyagent.max_volumes
    learned_policy = PolicyFromAgent(problem, chebyagent, mode='greedy')
    filename = 'weights' + str([str(scalarization_weights[i]) for i in xrange(len(scalarization_weights))])
    policy_heat_plot(problem, learned_policy, states, filename=filename)
    # just create two lists containing (0,0)
    u1 = [0]
    u1.extend(a_list)

    # extend longer list

    x = np.arange(len(u1))

    ##################################
    #               PLOT             #
    ##################################
    plt.figure()
    paretofront = [max(u1), ]*len(x)
    plt.title(problem.name())
    plt.plot(x, u1, 'r', label="Chebyshev-Agent")
    plt.plot(x, paretofront, 'g--', label="Paretofront")
    plt.legend(loc='lower right', frameon=False)
    plt.axis([0-0.01*len(u1), len(u1), 0, 1.1*max(u1)])
    plt.xlabel('interactions')
    plt.ylabel('hypervolume')
    plt.grid(True)
    plt.show()
