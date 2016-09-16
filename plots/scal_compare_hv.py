from morlbench.morl_problems import MORLGridworld
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot, policy_plot2
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':

    random.seed(3)
    np.random.seed(3)
    # create Problem
    problem = MORLGridworld()
    scalarization_weights = [0.0, ] * problem.reward_dimension
    # tau is for chebyshev agent
    tau = 1.0
    # ref point is used for Hypervolume calculation
    ref = [-0.1, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.03
    # Propability of epsilon greedy selection
    eps = 0.9
    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau)
    # both agents interact (times):
    interactions = 1500
    n_vectors = 2

    # weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]

    weights = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
               [0.33, 0.33, 0.33]]
    # weights = [[0.0, 1.0, 0.0]]
    linagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                    ref_point=ref, tau=tau, function='linear')
    hvb_hypervolumes = []
    cheb_hypervolumes = []
    wreward = []
    wreward2 = []
    for i in xrange(len(weights)):

        linagent._w, chebyagent._w = weights[i], weights[i]
        payouts1, moves1, states1 = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                    max_episode_length=150)
        payouts2, moves2, states2 = morl_interact_multiple_episodic(linagent, problem, interactions,
                                                                    max_episode_length=150)
        print weights[i]
        learned_policy = PolicyFromAgent(problem, linagent, mode='greedy')
        learned_policy1 = PolicyFromAgent(problem, chebyagent, mode='greedy')
        # policy_plot2(problem, learned_policy)
        # policy_heat_plot(problem, learned_policy, states1, title='chebishev')
        # policy_plot2(problem, learned_policy1)
        # policy_heat_plot(problem, learned_policy1, states2, title='linear')
        wreward.append(np.dot(weights[i], np.mean(payouts1, axis=0)))
        # print np.mean(payouts1, axis=0)
        wreward2.append(np.dot(weights[i], np.mean(payouts2, axis=0)))
        # print np.mean(payouts2, axis=0)
        #plot_hypervolume([chebyagent, linagent], problem)
        hvb_hypervolumes.append(max(linagent.max_volumes))
        cheb_hypervolumes.append(max(chebyagent.max_volumes))
        linagent.reset()
        chebyagent.reset()
    mpl.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    size = 0.48 * 5.8091048611149611602
    fig = plt.figure(figsize=[size, 0.75 * size])

    fig.set_size_inches(size, 0.7 * size)
    ax = fig.add_subplot(111)
    width = 0.3
    x = np.arange(1, len(weights)+1)
    ax.bar(x-width, hvb_hypervolumes, width, color='r', label="Linear", zorder=2)
    ax.bar(x, cheb_hypervolumes, width, color='b', label='Chebyshev', zorder=2)

    plt.axis([0-width, len(weights)+1, 0, 1.1*max([max(cheb_hypervolumes), max(hvb_hypervolumes)])])
    plt.xlabel('weights', size=8)
    plt.ylabel('hypervolume maximum', size=8)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=8)
    weights = [[round(weights[q][i], 2) for i in xrange(len(weights[q]))] for q in xrange(len(weights))]
    ax.set_xticks(x)
    labels = xrange(1, len(weights)+1,1)
    ax.set_xticklabels(labels, size=8)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.grid(True, zorder=0)
    plt.subplots_adjust(bottom=0.16, left=0.21, top=0.88)
    plt.show()
    # plt.savefig('comparison')
    fig = plt.figure(figsize=[size, 0.75 * size])

    fig.set_size_inches(size, 0.7 * size)
    ax = fig.add_subplot(111)
    width = 0.3
    x = np.arange(1, len(weights) + 1)
    ax.bar(x - width, wreward, width, color='r', label="Linear", zorder=2)
    ax.bar(x, wreward2, width, color='b', label='Chebyshev', zorder=2)

    plt.axis([0 - width, len(weights) + 1, 0, 1.1 * max([max(wreward2), max(wreward)])])
    plt.xlabel('weights', size=8)
    plt.ylabel('weighted average reward', size=8)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=8)
    weights = [[round(weights[q][i], 2) for i in xrange(len(weights[q]))] for q in xrange(len(weights))]
    ax.set_xticks(x)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    labels = xrange(1, len(weights)+1,1)
    ax.set_xticklabels(labels, size=8)
    print weights
    plt.grid(True, zorder=0)
    plt.subplots_adjust(bottom=0.16, left=0.21, top=0.88)
    plt.show()
    #plt.savefig('comparison')
