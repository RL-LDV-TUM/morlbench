#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 28, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_agents_multiple_crit import MORLConvexHullValueIteration
from morlbench.morl_problems import MORLResourceGatheringProblem, MORLGridworld, MORLBuridansAssProblem, Deepsea
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_plot2
import logging as log
import morlbench.progressbar as pgbar
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    max_interactions = 50
    problem = MORLResourceGatheringProblem()
    agent = MORLConvexHullValueIteration(problem)
    log.info('Playing maximum %i interactions each... ' % max_interactions)
    pbar = pgbar.ProgressBar(widgets=['Interactions: ', pgbar.SimpleProgress('/'), ' (', pgbar.Percentage(), ') ',
                                      pgbar.Bar(), ' ', pgbar.ETA()], maxval=max_interactions)
    pbar.start()
    for i in xrange(max_interactions):
        # update Q:
        for s in xrange(problem.n_states):
            for a in xrange(problem.n_actions):
                state = s
                action = a
                new_hull = [[0, ]*problem.reward_dimension]
                for s_prime in xrange(problem.n_states):
                    prob = problem.P[s, a, s_prime]
                    reward = problem.R[s_prime]
                    if reward.any():
                        pass
                    candidate = prob * (agent.scalar_multiplication(agent._V[s_prime], agent._gamma) + reward)[0]
                    new_hull = agent.vector_add(new_hull, candidate)
                agent._Q_sets[s, a] = [[0, ] * problem.reward_dimension]
                agent._Q_sets[s, a] = new_hull
        for s in xrange(problem.n_states):
            candidates = []
            for a in xrange(problem.n_actions):
                for p in xrange(len(agent._Q_sets[s, a])):
                    candidates.append(np.array(agent._Q_sets[s, a][p]))
            candidates = agent.hv_calculator.extract_front(candidates)
            if len(candidates) < problem.reward_dimension+1:
                for u in candidates:
                    np.append(agent._V[s], u)
            else:
                agent._V[s] = [candidates[x] for x in ConvexHull(candidates).simplices]
        pbar.update(i)

    weight = [0.0, 1.0]
    agent.extract_policy(weight)
    policy = PolicyFromAgent(problem, agent, mode='greedy')
    policy_plot2(problem, policy, 'w:0.0,1.0')
    plt.show()

    weight = [1.0, 1.0]
    agent.extract_policy(weight)
    policy = PolicyFromAgent(problem, agent, mode='greedy')
    policy_plot2(problem, policy, 'w:1.0,0.0')
    plt.show()

    weight = [0.7, 0.3]
    agent.extract_policy(weight)
    policy = PolicyFromAgent(problem, agent, mode='greedy')
    policy_plot2(problem, policy, 'w:0.7,0.3')
    plt.show()



