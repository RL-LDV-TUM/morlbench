from morlbench.morl_problems import MountainCarTime, Deepsea, MORLGridworld, MountainCar
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.plotting_stuff import plot_hypervolume
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create Problem
    problem = MountainCar(acc_fac=0.004, cf=0.0025, time_lim=90, state=76)
    # create an initialize randomly a weight vector
    scalarization_weights = [1.0, 0.0, 0.0]
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.5
    # Propability of epsilon greedy selection
    eps = 0.9

    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau, function='chebishev', gamma=0.9)
    hvbagent = MORLHVBAgent(problem, alfacheb, eps, ref, [1.0, 0.0])

    # both agents interact (times):
    interactions = 100
    #
    payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=400)
    print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
           str(states[:]) + '\n')
    #, moves, states = morl_interact_multiple_average_episodic(chebyagent, problem, 10, 500)

    # time = problem.time_token
    plot_hypervolume([chebyagent], problem)
    # state_frequency = np.zeros(problem.n_states)
    # for mov in moves:
    #     for st in xrange(problem.n_states):
    #         state_frequency[st] += mov.count(st)
    #
    f, axarr = plt.subplots(2, sharex=True)
    velocity = [0]
    for i in xrange(len(states[-1])-1):
        velocity.append(states[-1][i+1] - states[-1][i])
    for t in xrange(abs(len(states[-1])-len(velocity))):
        velocity.append(velocity[t-1])
    xv = np.arange(0, len(velocity))
    axarr[0].plot(xv, velocity, 'y', label='velocity')
    x = np.arange(0, len(states[-1]))
    axarr[1].plot(x, states[-1], 'b', label="states")
    y = np.zeros(len(states[-1]))
    goal = np.zeros(len(states[-1]))
    left_front = np.zeros(len(states[-1]))
    y[:] = 76
    goal[:] = problem.n_states
    left_front[:] = 0
    plt.plot(x, y, 'm--', label='Minimum')
    plt.axis([-1, 1.1*len(states[-1]), -2, 1.1*max(states[-1])])
    plt.plot(x, goal, 'g--', label='goal')
    plt.plot(x, left_front, 'r--', label='left_front')
    axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # plt.xlabel('time')
    # plt.ylabel('states visited')
    # x = np.arange(0, len(state_frequency))
    # plt.bar(x, state_frequency, 1.0, 'r', label='state visited')
    plt.show()
