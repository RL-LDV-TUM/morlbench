#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 28, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_agents_multiple_criteria import MORLConvexHullValueIteration
from morlbench.morl_problems import MORLResourceGatheringProblem, MORLGridworld, MORLBuridansAssProblem, Deepsea, MOPuddleworldProblem
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_plot2
import logging as log
import morlbench.progressbar as pgbar
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from morlbench.helpers import remove_duplicates

if __name__ == '__main__':

    max_interactions = 100
    problem = MORLBuridansAssProblem()
    agent = MORLConvexHullValueIteration(problem,gamma=0.1)
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
                new_hull = [[0, ] * problem.reward_dimension]
                for s_prime in xrange(problem.n_states):
                    prob = problem.P[s, a, s_prime]
                    if not prob:
                        continue
                    reward = problem.R[s_prime]
                    if reward.any():
                        pass
                    v = agent.scalar_multiplication(agent._V[s_prime], agent._gamma)
                    candidate = agent.scalar_multiplication(agent.vector_add(v, reward), prob)
                    new_hull = agent.hull_add(new_hull, candidate)
                try:
                    agent._Q_sets[s, a] = np.zeros_like(new_hull)
                except TypeError:
                    pass
                agent._Q_sets[s, a] = agent.hull_add(agent._Q_sets[s, a], new_hull)

        for s in xrange(problem.n_states):
            candidates = []
            for a in xrange(problem.n_actions):
                for p in xrange(len(agent._Q_sets[s, a])):
                    candidates.append(np.array(agent._Q_sets[s, a][p]))

            candidates = agent.hv_calculator.extract_front(candidates)
            candidates = remove_duplicates(candidates)
            domination = False
            k=[]
            for dim in xrange(len(candidates[0])):
                for u in xrange(len(candidates)):
                    if candidates[u][dim] == candidates[0][dim]:
                        k.append(candidates[u][dim])
                if len(k) == len(candidates):
                    domination = True
            if len(candidates) < problem.reward_dimension+1 or domination:
                for u in candidates:
                    agent._V[s] = agent.vector_add(agent._V[s], u)
            else:
                Hull = ConvexHull(candidates)
                vertices = Hull.vertices
                agent._V[s] = [candidates[x] for x in vertices]
        pbar.update(i)
    print agent._Q_sets
    # problem.n_states = 25
    weight = [0.0, 0.0, 1.0]
    agent.extract_policy(weight)
    policy = PolicyFromAgent(problem, agent, mode='greedy')
    policy_plot2(problem, policy, str(weight))
    plt.show()

    weight = [1.0, 0.0, 0.0]
    agent.extract_policy(weight)
    policy = PolicyFromAgent(problem, agent, mode='greedy')
    policy_plot2(problem, policy, str(weight))
    plt.show()

    weight = [0.7, 0.3, 0.0]
    agent.extract_policy(weight)
    policy = PolicyFromAgent(problem, agent, mode='greedy')
    policy_plot2(problem, policy, str(weight))
    plt.show()



