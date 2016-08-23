from morlbench.morl_problems import MountainCarTime, Deepsea, MORLGridworld, MountainCar
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.plotting_stuff import plot_hypervolume
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic

import numpy as np
import random
import matplotlib.pyplot as plt
import collections

if __name__ == '__main__':
    # we need this to show if accelerations have changed
    def mean_continued(data):
        mean_cont = [data[0]]
        for ind in xrange(1, len(data)):
            mean_cont.append((data[ind]+mean_cont[len(mean_cont)-1])/2.0)
        return mean_cont

    # create Problem
    problem = MountainCar(acc_fac=0.008, cf=0.0025)
    # create an initialize randomly a weight vector
    scalarization_weights = [1.0, 0.0, 0.0]
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.3
    # Propability of epsilon greedy selection
    eps = 0.9
    # should we show total acceleration count or just trend:
    show_trend = False

    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau, function='chebishev', gamma=1.0)
    hvbagent = MORLHVBAgent(problem, alfacheb, eps, ref, [1.0, 0.0])

    # both agents interact (times):
    interactions = 20
    #
    payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=300, discounted_eps=True)
    # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
    #       str(states[:]) + '\n')
    #, moves, states = morl_interact_multiple_average_episodic(chebyagent, problem, 10, 500)

    # time = problem.time_token
    states = problem.create_plottable_states(states)
    plot_hypervolume([chebyagent], problem)
    forward_acc = []
    backward_acc = []
    nothin = []
    for i in xrange(len(moves)):
        counter = list(moves[i])
        nothin.append(counter.count(0))
        forward_acc.append(counter.count(1))
        backward_acc.append(counter.count(2))
    x = np.arange(len(nothin))
    if show_trend:
        nothin = mean_continued(nothin)
        backward_acc = mean_continued(backward_acc)
        forward_acc = mean_continued(forward_acc)
    plt.plot(x, nothin, 'y', label='no_acceleration')
    plt.plot(x, forward_acc, 'g', label='forward acceleration')
    plt.plot(x, backward_acc, 'r', label='backward acceleration')
    plt.xlabel('epoch')
    plt.ylabel('count')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('count')
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
    y[:] = -0.5
    goal[:] = problem._goalxState
    left_front[:] = -1.5
    plt.plot(x, y, 'm--', label='Minimum')
    plt.axis([-1, 1.1*len(states[-1]), -1.55, 0.6])
    plt.plot(x, goal, 'g--', label='goal')
    plt.plot(x, left_front, 'r--', label='left_front')
    axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # plt.xlabel('time')
    # plt.ylabel('states visited')
    # x = np.arange(0, len(state_frequency))
    # plt.bar(x, state_frequency, 1.0, 'r', label='state visited')
    # plt.savefig(filename='mc')
    plt.show()
