from morlbench.morl_problems import MORLMountainCar
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.plotting_stuff import plot_hypervolume
from morlbench.experiment_helpers import morl_interact_multiple_episodic

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
                                      ref_point=ref, tau=tau, function='chebishev')
    hvbagent = MORLHVBAgent(problem, alfacheb, eps, ref, [1.0, 0.0])

    # both agents interact (times):
    interactions = 10000

    payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=200)
    print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
           str(states[:]) + '\n')

    time = problem.time_token
    plot_hypervolume([chebyagent], problem)
    state_frequency = np.zeros(problem.n_states)
    for mov in moves:
        for st in xrange(problem.n_states):
            state_frequency[st] += mov.count(st)

    plt.subplot()
    x = np.arange(0, len(time))
    plt.plot(x, time, 'r', label="time token")
    plt.subplot()
    x = np.arange(0, len(state_frequency))
    plt.bar(x, state_frequency, 1.0, 'r', label='state visited')
    plt.show()
