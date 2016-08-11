#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 07, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
import numpy as np
import matplotlib.pyplot as plt
from morlbench.morl_agents_multiple_criteria import MultipleCriteriaH, MultipleCriteriaR
from morlbench.morl_problems import MORLBuridansAssProblem, MOPuddleworldProblem, MORLGridworld, MountainCarAcceleration, MORLResourceGatheringProblem, Deepsea
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot, policy_plot2, policy_plot
from morlbench.helpers import HyperVolumeCalculator
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

if __name__ == '__main__':

    # how many vectors do you want to train before?
    n_vectors = 100
    # which problem do you want to experiment on?
    problem = MORLGridworld()
    # this is the threshold that the difference of the weighted average reward of the untrained and trained policy
    # should top, to let the new trained policy be a 'better policy'
    deltah = 0.2
    deltar = 0.3
    # epsilon = propability that the agent choses a greedy action instead of a random action
    epsilon = 0.9
    # learning rate
    alfar = 0.2
    alfah = 0.65
    betar = 0.01
    # how many interactions should the action train per weight?
    interactions = 100000
    # how many episodes in one interactions should be taken? (if the problem gets in a terminal state, it will be
    # interrupted anyway (episodes = steps in the environment = actions)
    max_per_interaction = 200
    # count of final weighted average reward that don't differ from the last ones to interrupt and signify converge
    converging_criterium = 25
    ref = [-1.0, ]*problem.reward_dimension
    # we want to evaluate both policy set with hypervolume indicator
    hv_calculator = HyperVolumeCalculator(ref)
    # create agent
    # agent = MultipleCriteriaH(problem, n_vectors, delta, epsilon, alfa, interactions, max_per_interaction,
    #                          converging_criterium)
    agent = MultipleCriteriaR(problem, n_vectors, deltar, epsilon, alfar, betar,  interactions, max_per_interaction,
                              converging_criterium)
    agent2 = MultipleCriteriaH(problem, n_vectors, deltah, epsilon, alfah,  interactions, max_per_interaction,
                              converging_criterium)
    # start the training
    agent.weight_training()
    agent2.weight_training()
    hvs = []
    rho = [i for i in agent.rhos.values()]
    pf = []
    for i in xrange(len(rho)):
        pf.append(rho[i])
        hvs.append(hv_calculator.compute_hv(pf))
    x = np.arange(len(hvs))
    plt.figure()
    plt.plot(x, hvs)
    plt.xlabel('policy')
    plt.ylabel('weighted average reward')
    print "R-Learning found %i policy/-ies, H-Learning found %i policy/-ies" % (len(agent.pareto), len(agent2.pareto))
    plt.figure()
    if problem.reward_dimension == 2:
        x1 = [i[0] for i in agent.pareto]
        y1 = [j[1] for j in agent.pareto]
        x2 = [k[0] for k in agent2.pareto]
        y2 = [l[0] for l in agent2.pareto]
        plt.plot(x1, y1, 'bo')
        plt.plot(x2, y2, 'ro')
    if problem.reward_dimension == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        apx, apy, apz, upx, upy, upz = [], [], [], [], [], []
        for i in agent.pareto:
            apx.append(i[0])
            apy.append(i[1])
            apz.append(i[2])
            ax.scatter(apx, apy, apz, 'b')
        plt.show()

        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        for u in agent2.pareto:
            upx.append(u[0])
            upy.append(u[1])
            upz.append(u[2])
            ax2.scatter(upx, upy, upz, 'ro')
        plt.show()

    print 'R:'+str(hv_calculator.compute_hv(agent.pareto))
    print 'H:'+str(hv_calculator.compute_hv(agent2.pareto))
    # now you can choose a specific weight and train on it
    specific_weight = [1.0, 0.0, 0.0]
    mean_count = np.mean(agent.interactions_per_weight)
    mean_count2 = np.mean(agent2.interactions_per_weight)
    agent.train_one_weight(specific_weight)
    agent2.train_one_weight(specific_weight)
    print 'R-agent needed for weight: ' + str(specific_weight) + ' ' + \
          str(agent.interactions_per_weight[len(agent.interactions_per_weight)-1]) + \
          ' interactions. Average before was:' + str(mean_count)
    print 'H-agent needed for weight: ' + str(specific_weight) + ' ' + \
          str(agent2.interactions_per_weight[len(agent2.interactions_per_weight)-1]) + \
          ' interactions. Average before was:' + str(mean_count2)

    agent.plot_interaction_rhos(specific_weight)
    agent2.plot_interaction_rhos(specific_weight)
    policy = PolicyFromAgent(agent2.problem, agent2, mode='greedy')
    policy_plot2(problem, policy)
    # print policy._pi


