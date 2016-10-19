#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jun 28, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>

    Copyright (C) 2016  Simon Woelzmueller

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from morlbench.morl_agents_multiple_criteria import MORLConvexHullValueIteration
from morlbench.morl_problems import MORLResourceGatheringProblem, MORLGridworld, MORLBuridansAssProblem, Deepsea,\
        MOPuddleworldProblem, MountainCarTime
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_plot2
import logging as log
import morlbench.progressbar as pgbar
from pyhull.convex_hull import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from morlbench.helpers import remove_duplicates, compute_hull

if __name__ == '__main__':

    max_interactions = 40
    problem = MORLResourceGatheringProblem()
    agent = MORLConvexHullValueIteration(problem, gamma=0.9)
    log.info('Playing maximum %i interactions each... ' % max_interactions)
    pbar = pgbar.ProgressBar(widgets=['Interactions: ', pgbar.SimpleProgress('/'), ' (', pgbar.Percentage(), ') ',
                                      pgbar.Bar(), ' ', pgbar.ETA()], maxval=max_interactions)
    pbar.start()
    for i_count in xrange(max_interactions):
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
                    if (reward > 0).any():
                        pass
                    v = agent.scalar_multiplication(agent._V[s_prime], agent._gamma)
                    candidate = agent.scalar_multiplication(agent.vector_add(v, reward), prob)
                    new_hull = agent.hull_add(new_hull, candidate)
                new_hull = remove_duplicates(new_hull)
                agent._Q_sets[agent.s_a_mapping[s, a]] = np.zeros_like(new_hull)
                agent._Q_sets[agent.s_a_mapping[s, a]] = (agent.hull_add(agent._Q_sets[agent.s_a_mapping[s, a]], new_hull))

        for s in xrange(problem.n_states):
            candidates = []
            for a in xrange(problem.n_actions):
                for p in xrange(len(agent._Q_sets[agent.s_a_mapping[s, a]])):
                    candidates.append(np.array(agent._Q_sets[agent.s_a_mapping[s, a]][p]))

            candidates = remove_duplicates(candidates)
            # candidates = agent.hv_calculator.extract_front(candidates)
            candidates = agent.get_hull(candidates)
            agent._V[s] = candidates
        pbar.update(i_count)
    print agent._Q_sets
    # problem.n_states = 25
    if problem.reward_dimension == 3:
        weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5],
               [0.5, 0.0, 0.5], [0.33, 0.33, 0.33]]
    if problem.reward_dimension == 2:
        weights = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    for weight in weights:
        agent.extract_policy(weight)

        policy = PolicyFromAgent(problem, agent, mode='greedy')
        policy_plot2(problem, policy, str(weight))
        plt.show()
