#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mai 25 2015

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_problems import MOPuddleworldProblem, MORLBurdiansAssProblem, MORLGridworld, Deepsea
from morlbench.morl_agents import  MORLHLearningAgent
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_heat_plot, policy_plot2


import numpy as np
import random
import matplotlib.pyplot as plt

"""
This experiment should show the mechanism of variing the criteria to improve an Agent
Recommended Agents: HLearningAgent, RLearning Agent
"""
if __name__ == '__main__':

    # count of weight vevtors
    n_vectors = 20
    # create problem
    problem = MORLBurdiansAssProblem()
    # 6 agents with each different weights
    weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]
    # threshold an policy is good
    delta = 1.0
    # for epsilon greedy action selection
    epsilon = 0.3
    # learning rate
    alfa = 0.6
    # agents first construction (first weight will be ignored
    hlearning = MORLHLearningAgent(problem, epsilon, alfa, [0.1, ]*problem.reward_dimension)
    # storage
    policies = []
    rewards = []
    rhos = []
    hs = []
    weighted_list = []
    # every weight vector will be used
    for i in xrange(len(weights)):
        # put it into the agent
        hlearning.w = weights[i]
        # if there are any stored policies
        if policies:
            # look for the best
            weighted = [np.dot(weights[u], rhos[u]) for u in xrange(len(rhos))]
            piopt = weighted.index(max(weighted))
            weighted_list.append(max(weighted))
            print(weighted[piopt])
            # put its parameters back into the agent
            hlearning._rho = rhos[piopt]
            hlearning._reward = rewards[piopt]
            hlearning._h = hs[piopt]
        # extract old rho vector
        old_rho = hlearning._rho

        problem.reset()
        # while the agent with new weights isn't better than the old one
        while np.dot(weights[i], (hlearning._rho-old_rho)) < delta:
            # get state of the problem
            last_state = problem.state
            # take next best action
            action = hlearning.decide(0, problem.state)
            # execute that action
            payout = problem.play(action)
            # obtain new state
            new_state = problem.state
            # learn from that action
            hlearning.learn(0, last_state, action, payout, new_state)
        # at the end, get the best policy
        policy = PolicyFromAgent(problem, hlearning, mode='greedy')
        # store it
        policies.append(policy)
        # and all that other stuff we need later
        rewards.append(hlearning._reward)
        rhos.append(hlearning._rho)
        hs.append(hlearning._h)
    # check for the best rho vector
    weighted = [np.dot(weights[u], rhos[u]) for u in xrange(len(rhos))]
    ###################################################################
    #       PLOT (Curve for Learning Process and Policy Plot)         #
    ###################################################################
    plt.figure()
    x = np.arange(len(weighted_list))
    plt.plot(x, weighted_list)
    plt.axis([0, 1.1*len(weighted_list), 0, max(weighted_list)])
    plt.show()
    piopt = weighted.index(max(weighted))
    policy_plot2(problem, policies[piopt])

