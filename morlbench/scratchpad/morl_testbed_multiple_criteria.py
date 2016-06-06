#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mai 25 2015

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_problems import MOPuddleworldProblem, MORLBurdiansAssProblem, MORLGridworld
from morlbench.morl_agents import MORLHLearningAgent, MORLRLearningAgent
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_heat_plot, policy_plot2


import numpy as np
# import random
import matplotlib.pyplot as plt
import logging as log
import morlbench.progressbar as pgbar


"""
This experiment should show the mechanism of variing the criteria to improve an Agent
Recommended Agents: HLearningAgent, RLearning Agent
"""


def multiple_criteria_h_learning(problem=None, n_vectors=20, delta=10.0,  epsilon=0.8, alfa=0.1,
                                 interactions=10000, max_per_interaction=150, converging_criterium=20):

    if problem is None:
        problem = MORLGridworld()
    # weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    #            [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.33, 0.33, 0.33]]
    weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]
    # agents first construction (first weight will be ignored
    hlearning = MORLHLearningAgent(problem, epsilon, alfa, [0.1, ]*problem.reward_dimension)
    # storage
    policies = dict()
    rewards = dict()
    rhos = dict()
    hs = dict()
    weighted_list = dict()
    interactions_per_weight = []
     # ------PROGRESSBAR START/ LOGGING -----------#
    log.info('Playing  %i interactions on %i vectors...', interactions, len(weights))
    pbar = pgbar.ProgressBar(widgets=['Weight vector: ', pgbar.SimpleProgress('/'), ' (', pgbar.Percentage(), ') ',
                             pgbar.Bar(), ' ', pgbar.ETA()], maxval=len(weights))
    pbar.start()
    # every weight vector will be used
    for i in xrange(len(weights)):
        # put it into the agent
        hlearning.w = weights[i]
        # if there are any stored policies
        if policies:
            # look for the best
            weighted = [np.dot(weights[u], rhos[u]) for u in rhos.iterkeys()]
            piopt = rhos.keys()[weighted.index(max(weighted))]
            weighted_list[piopt] = max(weighted)
            # print(weighted[weighted.index(max(weighted))])
            # put its parameters back into the agent
            hlearning._rho = rhos[piopt]
            hlearning._reward = rewards[piopt]
            hlearning._h = hs[piopt]
        # extract old rho vector
        old_rho = hlearning._rho
        interaction_rhos = []
        # play for interactions times:
        for t in xrange(interactions):

            # only for a maximum of epsiodes(without terminating problem)
            for actions in xrange(max_per_interaction):
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
                if problem.terminal_state:
                    break
            interaction_rhos.append(hlearning._rho)
            # check after some interactions if we have converged
            if t > converging_criterium:
                # pick the last phis
                last_twenty = interaction_rhos[t-converging_criterium-1:t-1]
                # pick the last one
                last_one = interaction_rhos[t-1]
                # create a list that compares all of the twenty with the last
                compare = np.array([(last_twenty[l] == last_one).all() for l in range(converging_criterium)])
                # if all are same, the algorithm seems to converge
                if compare.all():
                    # create vector of weighted average reward
                    interaction_rhos_plot = [np.dot(weights[i], interaction_rhos[r]) for r in xrange(len(interaction_rhos))]
                    plt.figure()
                    plt.axis([0, 1.1*len(interaction_rhos_plot), min(interaction_rhos_plot), 1.1*max(interaction_rhos_plot)])
                    x = np.arange(len(interaction_rhos_plot))
                    plt.plot(x, interaction_rhos_plot, label=str(weights[i]))
                    plt.legend(loc='lower right', frameon=False)
                    plt.show()
                    interactions_per_weight.append(t)
                    break

        # at the end, get the policy
        interactions_per_weight.append(t)
        policy = PolicyFromAgent(problem, hlearning, mode='greedy')
        if (np.dot(weights[i], hlearning._rho) - np.dot(weights[i], old_rho)) > delta:
            # store it
            policies[i] = policy
            # and all that other stuff we need later
            rewards[i] = hlearning._reward
            rhos[i] = hlearning._rho
            hs[i] = hlearning._h
        pbar.update(i)


    ###################################################################
    #       PLOT (Curve for Learning Process and Policy Plot)         #
    ###################################################################
    plt.figure()
    x = np.arange(len(interactions_per_weight))
    plt.plot(x, interactions_per_weight)
    plt.axis([0, 1.1*len(interactions_per_weight), 0, 1.1*max(interactions_per_weight)])
    plt.show()
    #policy_plot2(problem, policies[piopt])


def multiple_criteria_r_learning(n_vectors=100, epsilon=0.6, alfa=0.01, beta=0.1, delta=2.0, interactions=10000,
                                 max_per_interaction=150, converging_criterium=20):
    # create problem
    problem = MORLGridworld()
    # 6 agents with each different weights
    weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]
    # agents first construction (first weight will be ignored
    r_learning = MORLRLearningAgent(problem, epsilon, alfa, beta, [0.1, ]*problem.reward_dimension)
    # storage
    policies = dict()
    rewards = dict()
    rhos = dict()
    Rs = dict()
    weighted_list = dict()
    # ------PROGRESSBAR START/ LOGGING -----------#
    log.info('Playing  %i interactions on %i vectors...', interactions, len(weights))
    pbar = pgbar.ProgressBar(widgets=['Weight vector: ', pgbar.SimpleProgress('/'), ' (', pgbar.Percentage(), ') ',
                             pgbar.Bar(), ' ', pgbar.ETA()], maxval=len(weights))
    pbar.start()
    # every weight vector will be used
    for i in xrange(len(weights)):
        # put it into the agent
        r_learning.w = weights[i]
        # if there are any stored policies
        if policies:
            # look for the best
            weighted = [np.dot(weights[u], rhos[u]) for u in rhos.iterkeys()]
            piopt = rhos.keys()[weighted.index(max(weighted))]
            weighted_list[i](max(weighted))
            print(weighted[piopt])
            # put its parameters back into the agent
            r_learning._rho = rhos[piopt]
            r_learning._R = Rs[piopt]
        # extract old rho vector
        old_rho = r_learning._rho
        interaction_rhos = []
        problem.reset()
        for t in xrange(interactions):
            # while the agent with new weights isn't better than the old one
            for actions in xrange(max_per_interaction):
                # if problem.terminal_state:
                    # break
                # get state of the problem
                last_state = problem.state
                # take next best action
                action = r_learning.decide(0, problem.state)
                # execute that action
                payout = problem.play(action)
                # obtain new state
                new_state = problem.state
                # learn from that action
                r_learning.learn(0, last_state, action, payout, new_state)
            # append actual rho
            interaction_rhos.append(r_learning._rho)
            # check after some interactions if we have converged
            if t > converging_criterium:
                # pick the last phis
                last_twenty = interaction_rhos[t-converging_criterium-1:t-1]
                # pick the last one
                last_one = interaction_rhos[t-1]
                # create a list that compares all of the twenty with the last
                compare = np.array([last_twenty[l] == last_one for l in xrange(converging_criterium)])
                # if all are same, the algorithm seems to converge
                if compare.all():
                    break
            if (np.dot(weights[i], r_learning._rho) - np.dot(weights[i], old_rho)) > delta:
                policy = PolicyFromAgent(problem, r_learning, mode='greedy')
                # store it
                policies[i] = policy
                # and all that other stuff we need later
                rewards.append(r_learning._reward)
                rhos.append(r_learning._rho)
                Rs.append(r_learning._R)
        pbar.update(i)
    # check for the best rho vector
    weighted = [np.dot(weights[u], rhos[u]) for u in xrange(len(rhos))]
    ###################################################################
    #       PLOT (Curve for Learning Process and Policy Plot)         #
    ###################################################################
    plt.figure()
    x = np.arange(len(weighted_list))
    #rewards = np.dot(rewards, weights)

    plt.plot(x, weighted_list)
    plt.axis([0, 1.1*len(weighted_list), 0, max(weighted_list)])
    plt.show()
    piopt = weighted.index(max(weighted))
    #policy_plot2(problem, policies[piopt])

multiple_criteria_h_learning()