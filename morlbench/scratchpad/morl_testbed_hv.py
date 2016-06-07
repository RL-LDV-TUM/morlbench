from morlbench.morl_problems import  Deepsea, MOPuddleworldProblem, MORLBurdiansAssProblem, MORLGridworld, MORLResourceGatheringProblem
from morlbench.morl_agents import MORLChebyshevAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create Problem
    problem = MOPuddleworldProblem()
    # create an initialize randomly a weight vector
    scalarization_weights = np.zeros(problem.reward_dimension)
    scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)], len(scalarization_weights))
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-24.0, -1.0]
    # learning rate
    alf = 0.2
    alfacheb = 0.1
    alfahvb = 0.1

    # Propability of epsilon greedy selection
    eps = 0.1
    # create one agent using chebyshev scalarization method
    chebyagent = MORLChebyshevAgent(problem, [0.5, 0.5], alpha=alfacheb, epsilon=eps,
                                    tau=tau, ref_point=ref)
    # create one agent using Hypervolume based Algorithm
    hvbagent = MORLHVBAgent(problem, alpha=alfahvb, epsilon=0.6, ref=ref, scal_weights=[1.0, 10.0])
    # both agents interact (times):
    interactions = 1000
    # make the interactions
    payouts, moves, states = morl_interact_multiple(chebyagent, problem, interactions,
                                                    max_episode_length=150)
    # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
    #     str(states[:]) + '\n')

    payouts2, moves2, states2 = morl_interact_multiple(hvbagent, problem, interactions,
                                                       max_episode_length=150)
    # print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
    #      str(states2[:]) + '\n')

    # extract all volumes of each agent
    a_list = chebyagent.max_volumes
    v_list = hvbagent.max_volumes
    # solution = len(a_list)/self.interactions
    solution = 1
    # data preparation for another solution !=1
    if solution != 1:
        # nice curve contains (0,0)
        u1 = [0]
        # cut the longer list
        overlay = len(v_list) % solution
        if overlay:
            del a_list[len(a_list)-overlay:]
        z = 0
        # append mean values
        while z < len(a_list):
            u1.append(np.mean(a_list[z:z+solution]))
            z += solution
        # create x vector
        x = np.arange(((len(a_list)/solution)-len(a_list) % solution)+1)
        # solution = len(v_list)/self.interactions
        u2 = [0]
        # cut the longer list
        overlay = len(v_list) % solution

        if overlay:
            del v_list[len(v_list)-overlay:]
        z = 0
        while z < len(v_list):
            u2.append(np.mean(v_list[z:z+solution]))
            z += solution
    else:
        # just create two lists containing (0,0)
        u1 = [0]
        u1.extend(a_list)
        u2 = [0]
        u2.extend(v_list)

    # extend longer list
    if len(u2) > len(u1):
        for i in range(len(u2)-len(u1)):
            u1.append(u1[len(u1)-1])
    else:
        for i in range(len(u1)-len(u2)):
            u2.append(u2[len(u2)-1])

    x = np.arange(min([len(u1), len(u2)]))

    ##################################
    #               PLOT             #
    ##################################

    x1, y1 = u1.index(max(u1)), max(u1)
    x2, y2 = u2.index(max(u2)), max(u2)
    paretofront = [max([max(u1), max(u2)]), ]*len(x)
    plt.title(problem.name())
    plt.plot(x, u1, 'r', label="Chebyshev-Agent")
    plt.plot(x, u2, 'b', label="HVB-Agent")
    plt.plot(x, paretofront, 'g--', label="Paretofront")
    plt.legend(loc='lower right', frameon=False)
    plt.axis([0-0.01*len(u1), len(u1), 0, 1.1*max([max(u1), max(u2)])])
    plt.xlabel('interactions')
    plt.ylabel('hypervolume')
    plt.grid(True)
    plt.show()
