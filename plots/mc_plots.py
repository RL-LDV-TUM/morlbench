from morlbench.morl_problems import MountainCarTime, Deepsea, MORLGridworld, MountainCar
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.plotting_stuff import plot_hypervolume
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections

if __name__ == '__main__':
    mpl.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    random.seed(43)
    np.random.seed(43)
    # we need this to show if accelerations have changed

    def mean_continued(data):
        mean_cont = [data[0]]
        for ind in xrange(1, len(data)):
            mean_cont.append((data[ind]+mean_cont[len(mean_cont)-1])/2.0)
        return mean_cont

    # create Problem
    problem = MountainCar(acc_fac=0.007, cf=0.0030)
    # create an initialize randomly a weight vector
    scalarization_weights = [1.0, 0.0, 0.0]
    # tau is for chebyshev agent
    tau = 1.0
    # ref point is used for Hypervolume calculation
    ref = [-10.0, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.4
    # Propability of epsilon greedy selection
    eps = 0.9
    # should we show total acceleration count or just trend:
    show_trend = True

    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau, gamma=0.9)
    # hvbagent = MORLHVBAgent(problem, alfacheb, eps, ref, [0.0, 0.0])

    # agent interact (times):
    interactions = 200
    #
    payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=300, discounted_eps=False)
    # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
    #       str(states[:]) + '\n')
    #, moves, states = morl_interact_multiple_average_episodic(chebyagent, problem, 10, 500)

    # time = problem.time_token
    chebyagent._epsilon = 1.0
    payouts, moves2, states = morl_interact_multiple_episodic(chebyagent, problem, 1, 300)
    velocity = problem.get_velocities(states)
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
    # plt.savefig('count')
    size = 0.48 * 5.8091048611149611602
    fig, axarr = plt.subplots(2, sharex=True, figsize=[size, 0.75 * size])
    fig.set_size_inches(size, 0.7 * size)
    xv = np.arange(0, len(velocity[-1]))
    axarr[0].plot(xv, velocity[-1], 'r', label='velocity')
    x = np.arange(0, len(states[-1]))
    axarr[1].plot(x, states[-1], 'b', label="states")

    y = np.zeros(len(states[-1]))
    goal = np.zeros(len(states[-1]))
    left_front = np.zeros(len(states[-1]))
    y[:] = -0.5
    goal[:] = problem._goalxState
    left_front[:] = -1.2
    axarr[1].plot(x, y, 'm--', label='minimum(start state)')
    axarr[1].axis([-1, 1.1*len(states[-1]), -1.25, 0.6])
    axarr[1].plot(x, goal, 'g--', label='goal position')
    axarr[1].plot(x, left_front, 'r--', label='left wall')
    # axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    axarr[1].set_xlabel('Epoche', size=9)
    axarr[1].set_ylabel('x', size=9)
    axarr[0].set_ylabel('v', size=9)
    plt.setp(axarr[1].get_yticklabels(), size=8)
    plt.setp(axarr[0].get_yticklabels(), size=8)
    plt.setp(axarr[1].get_xticklabels(), size=8)

    plt.subplots_adjust(hspace=0.12, bottom=0.19, left=0.22)
    plt.show()
