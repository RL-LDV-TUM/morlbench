from morlbench.morl_problems import MORLResourceGatheringProblem, MountainCarTime, MORLGridworld, MORLBuridansAssProblem, Deepsea
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot
from morlbench.plotting_stuff import plot_hypervolume
from morlbench.helpers import HyperVolumeCalculator

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create Problem
    problem = MORLGridworld()
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
    hv_calc = HyperVolumeCalculator(ref)
    # create one agent using chebyshev scalarization method
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                      ref_point=ref, tau=tau)

    linearagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfacheb, scalarization_weights=scalarization_weights,
                                       ref_point=ref, tau=tau, function='linear')
    # both agents interact (times):
    interactions = 1000

    c_payouts, c_moves, c_states = morl_interact_multiple_episodic(chebyagent, problem, interactions,
                                                                 max_episode_length=150)

    l_payouts, l_moves, l_states = morl_interact_multiple_episodic(linearagent, problem, interactions,
                                                                 max_episode_length=150)

    c_rewards = []
    for i in xrange(len(c_payouts)):
        cummulated = np.zeros(problem.reward_dimension)
        for u in xrange(i, 0, -1):
            cummulated += c_payouts[u]

        c_rewards.append(cummulated)
    l_rewards = []
    for k in xrange(len(l_payouts)):
        cummulated = np.zeros(problem.reward_dimension)
        for l in xrange(k, 0, -1):
            cummulated += l_payouts[l]

        l_rewards.append(cummulated)
    l_hv, c_hv = [], []

    for i in xrange(0, min([len(l_rewards), len(c_rewards)])):
        l_hv.append(hv_calc.compute_hv(l_rewards[:i]))
        c_hv.append(hv_calc.compute_hv(c_rewards[:i]))
        r = i

    x = np.arange(len(l_hv))
    plt.figure()
    plt.plot(x, l_hv)
    x = np.arange(len(c_hv))
    plt.plot(x, c_hv)
    plt.show()



