#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 07, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
import numpy as np
from morlbench.morl_agents_multiple_crit import MultipleCriteriaH
from morlbench.morl_problems import MORLBuridansAssProblem, MOPuddleworldProblem
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot, policy_plot2, policy_plot

if __name__ == '__main__':

    # how many vectors do you want to train before?
    n_vectors = 20
    # which problem do you want to experiment on?
    problem = MOPuddleworldProblem()
    # this is the threshold that the difference of the weighted average reward of the untrained and trained policy
    # should top, to let the new trained policy be a 'better policy'
    delta = 1.0
    # epsilon = propability that the agent choses a greedy action instead of a random action
    epsilon = 0.2
    # learning rate
    alfa = 0.1
    # how many interactions should the action train per weight?
    interactions = 100000
    # how many episodes in one interactions should be taken? (if the problem gets in a terminal state, it will be
    # interrupted anyway (episodes = steps in the environment = actions)
    max_per_interaction = 150
    # count of final weighted average reward that don't differ from the last ones to interrupt and signify converge
    converging_criterium = 20
    # create agent
    agent = MultipleCriteriaH(problem, n_vectors, delta, epsilon, alfa, interactions, max_per_interaction,
                              converging_criterium)

    # start the training
    agent.weight_training()

    # now you can choose a specific weight and train on it
    specific_weight = [0.0, 1.0]
    mean_count = np.mean(agent.interactions_per_weight)
    agent.train_one_weight(specific_weight)
    print 'agent needed for weight: ' + str(specific_weight) + ' ' + \
          str(agent.interactions_per_weight[len(agent.interactions_per_weight)-1]) + \
          ' interactions. Average before was:' + str(mean_count)

    agent.plot_interaction_rhos(specific_weight)
    policy = PolicyFromAgent(agent.problem, agent, mode='greedy')
    policy_plot2(problem, policy)
    print policy._pi


