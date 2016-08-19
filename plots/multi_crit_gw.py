#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 07, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
import numpy as np
import matplotlib.pyplot as plt
from morlbench.morl_agents_multiple_criteria import MultipleCriteriaH, MultipleCriteriaR
from morlbench.morl_problems import MORLBuridansAssProblem, MOPuddleworldProblem, MORLGridworld, MountainCarTime,\
    MORLResourceGatheringProblem, Deepsea, MORLBuridansAss1DProblem
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot, policy_plot2, policy_plot
from morlbench.helpers import HyperVolumeCalculator
import random
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

if __name__ == '__main__':

    # how many vectors do you want to train before?
    n_vectors = 100
    # np.random.seed(2)
    # random.seed(3)
    # which problem do you want to experiment on?
    problem = MOPuddleworldProblem(16)
    # this is the threshold that the difference of the weighted average reward of the untrained and trained policy
    # should top, to let the new trained policy be a 'better policy'
    deltah = 0.6
    # epsilon = propability that the agent choses a greedy action instead of a random action
    epsilon = 0.2
    # learning rate
    alfah = 0.65
    # how many interactions should the action train per weight?
    interactions = 1000
    # how many episodes in one interactions should be taken? (if the problem gets in a terminal state, it will be
    # interrupted anyway (episodes = steps in the environment = actions)
    max_per_interaction = 300
    # count of final weighted average reward that don't differ from the last ones to interrupt and signify converge
    converging_criterium = 25
    ref = [-0.001, ]*problem.reward_dimension
    # we want to evaluate both policy set with hypervolume indicator
    hv_calculator = HyperVolumeCalculator(ref)
    # create agent
    # agent = MultipleCriteriaH(problem, n_vectors, delta, epsilon, alfa, interactions, max_per_interaction,
    #                          converging_criterium)
    agent = MultipleCriteriaH(problem, n_vectors, deltah, epsilon, alfah, interactions, max_per_interaction,
                               converging_criterium)

    # start the training
    agent.weight_training()

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
    plt.title('H_Learning Average Reward Evolution')
    print 'max. HV' + str(max(hvs))
    plt.xlabel('policy')
    plt.ylabel('weighted average reward')
    print "H-Learning found %i policy/-ies" % (len(agent.pareto))
    plt.figure()
    if problem.reward_dimension == 2:
        x1 = [i[0] for i in agent.pareto]
        y1 = [j[1] for j in agent.pareto]

        plt.plot(x1, y1, 'bo')

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

    print 'R:'+str(hv_calculator.compute_hv(agent.pareto))
    # now you can choose a specific weight and train on it
    weights = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    mean_count = np.mean(agent.interactions_per_weight)
    interacts = []
    i = 0
    for specific_weight in weights:
        agent.train_one_weight(specific_weight)
        interacts.append(agent.interactions_per_weight[len(agent.interactions_per_weight)-1])
        print 'H-agent needed for weight: ' + str(specific_weight) + ' ' + \
              str(interacts[i]) + \
              ' interactions. Average before was:' + str(mean_count)
        i += 1
        agent.plot_interaction_rhos(specific_weight)

        policy2 = PolicyFromAgent(agent.problem, agent, mode='greedy')
        policy_plot2(problem, policy2)

    print 'Average interactions needed for the specific weights: ' + str(np.mean(interacts))

