#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mai 25 2015

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""
from morlbench.morl_problems import MOPuddleworldProblem, MORLBuridansAssProblem, MORLGridworld, Deepsea
from morlbench.morl_agents import MORLHLearningAgent, MORLRLearningAgent
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_plot, transition_map, heatmap_matplot, policy_heat_plot, policy_plot2


import numpy as np
# import random
import matplotlib.pyplot as plt
import logging as log
import morlbench.progressbar as pgbar


"""
These agents are used for multiple criteria average reward experiments
"""


class MultipleCriteriaH:
    """
    This class uses HLearning and a bunch of weight vectors to iterate through. After learning all of them,
    using the agent for a specific weight would cause faster performance and converging average reward
    """
    def __init__(self, problem=None, n_vectors=55, delta=1.0,  epsilon=0.1, alfa=0.01,
                                     interactions=100000, max_per_interaction=150, converging_criterium=60):
        if problem is None:
            self.problem = MORLGridworld()
        else:
            self.problem = problem
        # weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        #            [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.33, 0.33, 0.33]]
        self.weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]
        # agents first construction (first weight will be ignored
        self.hlearning = MORLHLearningAgent(problem, epsilon, alfa, [0.1, ]*problem.reward_dimension)
        # storage
        self.converged = False
        self.interactions = interactions
        self.delta = delta
        self.max_per_interaction = max_per_interaction
        self.converging_criterium = converging_criterium
        self.policies = dict()
        self.rewards = dict()
        self.rhos = dict()
        self.hs = dict()
        self.weighted_list = dict()
        self.interactions_per_weight = []
        self.stored = dict()
        self.old_rho = np.zeros(self.problem.reward_dimension)
        self.interaction_rhos = []

    def get_learned_action(self, state):
        return self.hlearning.get_learned_action(state)

    def weight_training(self):
        """
        takes n vectors and trains the agent till his weighted average reward converges
        :return:
        """
        # ------PROGRESSBAR START/ LOGGING -----------#
        log.info('Playing  %i interactions on %i vectors...', self.interactions, len(self.weights))
        pbar = pgbar.ProgressBar(widgets=['Weight vector: ', pgbar.SimpleProgress('/'), ' (', pgbar.Percentage(), ') ',
                                 pgbar.Bar(), ' ', pgbar.ETA()], maxval=len(self.weights))
        pbar.start()
        # every weight vector will be used
        for i in xrange(len(self.weights)):
            # put it into the agent
            self.train_one_weight(self.weights[i])
            # plot the evolution of rhos
            self.plot_interaction_rhos(self.weights[i])
            # evaluate and store policy if good. a True is stored in self.stored[i] if the policy was good enough
            self.stored[i] = self.evaluate_new_policy(self.old_rho, i)
            pbar.update(i)

        self.plot_interactions_per_weight()
        return True

    def plot_interactions_per_weight(self):
        ###################################################################
        #       PLOT (Curve for Learning Process and Policy Plot)         #
        ###################################################################
        fig = plt.figure()
        x = np.arange(len(self.interactions_per_weight))
        plt.plot(x, self.interactions_per_weight, label="interactios per weight")
        for i in range(len(self.stored)):
            if self.stored[i]:
                plt.axvline(i, color='r', linestyle='--')
        plt.axis([0, 1.1*len(self.interactions_per_weight), 0, 1.1*max(self.interactions_per_weight)])
        plt.xlabel("weight count")
        plt.ylabel("count of interactions ")
        plt.draw()

    def look_for_opt(self, weight):
        weighted = [np.dot(weight, self.rhos[u]) for u in self.rhos.iterkeys()]
        max_weighted = max(weighted)
        index_max = weighted.index(max_weighted)
        piopt = self.rhos.keys()[index_max]
        return piopt, weighted

    def evaluate_new_policy(self, old_rho, i):
        """
        this function takes the learned policy and compares the weighted average reward with the same policy
        before learning, if it is better, it stores the new agents params into a pool of "good policies"
        :param old_rho: weighted average reward of policy i before the learning process
        :param i: policy number
        :return:
        """
        policy = PolicyFromAgent(self.problem, self.hlearning, mode='greedy')
        print np.dot(self.weights[i], self.hlearning._rho) - np.dot(self.weights[i], old_rho)
        if np.abs(np.dot(self.weights[i], self.hlearning._rho) - np.dot(self.weights[i], old_rho)) > self.delta:
            # store it
            self.policies[i] = policy
            # and all that other stuff we need later
            self.rewards[i] = self.hlearning._reward
            self.rhos[i] = self.hlearning._rho
            self.hs[i] = self.hlearning._h
            return True
        else:
            return False

    def plot_interaction_rhos(self, weight):
        interaction_rhos_plot = [np.dot(weight, self.interaction_rhos[r]) for r in xrange(len(self.interaction_rhos))]
        plt.figure()
        plt.axis([0, 1.1*len(interaction_rhos_plot), -1.1*np.abs(min(interaction_rhos_plot)), 1.1*max(interaction_rhos_plot)])
        x = np.arange(len(interaction_rhos_plot))
        plt.plot(x, interaction_rhos_plot, label=str(weight)+' converged: '+str(self.converged))
        plt.xlabel("interactions at this weight")
        plt.ylabel("weighted average reward")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
        self.converged = False

    def train_one_weight(self, weight):
        if len(weight) != self.problem.reward_dimension:
            log.info("could not train this weight, wrong dimension")
            return
        else:
            self.hlearning.w = weight
            # if there are any stored policies
            if self.policies:
                # look for the best
                piopt, weighted = self.look_for_opt(weight)
                self.weighted_list[piopt] = max(weighted)
                # print(weighted[weighted.index(max(weighted))])
                # put its parameters back into the agent
                self.hlearning._rho = self.rhos[piopt]
                self.hlearning._reward = self.rewards[piopt]
                self.hlearning._h = self.hs[piopt]
            # extract old rho vector
            self.old_rho = self.hlearning._rho
            self.interaction_rhos = []
            # play for interactions times:
            for t in xrange(self.interactions):

                # only for a maximum of epsiodes(without terminating problem)
                for actions in xrange(self.max_per_interaction):
                    # get state of the problem
                    last_state = self.problem.state
                    # take next best action
                    action = self.hlearning.decide(0, self.problem.state)
                    # execute that action
                    payout = self.problem.play(action)
                    # obtain new state
                    new_state = self.problem.state
                    # learn from that action
                    self.hlearning.learn(0, last_state, action, payout, new_state)
                    if self.problem.terminal_state:
                        break
                self.interaction_rhos.append(self.hlearning._rho)
                # check after some interactions if we have converged
                if t > self.converging_criterium:
                    # pick the last phis

                    last_twenty = self.interaction_rhos[t-self.converging_criterium-1:t-1]
                    # pick the last one
                    last_one = self.interaction_rhos[t-1]
                    # create a list that compares all of the twenty with the last
                    compare = np.array([(last_twenty[l] == last_one).all() for l in range(self.converging_criterium)])
                    # if all are same, the algorithm seems to converge
                    if compare.all():
                        self.converged = True
                        break
            # store the count of interactions to show convergence acceleration
            self.interactions_per_weight.append(t)


class MultipleCriteriaR:
    """
    This class uses HLearning and a bunch of weight vectors to iterate through. After learning all of them,
    using the agent for a specific weight would cause faster performance and converging average reward
    """
    def __init__(self, problem=None, n_vectors=55, delta=1.0,  epsilon=0.1, alfa=0.01, beta=1.0,
                 interactions=100000, max_per_interaction=150, converging_criterium=60, weights = None):
        if problem is None:
            self.problem = MORLGridworld()
        else:
            self.problem = problem
        if weights is None:
            # weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            #            [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.33, 0.33, 0.33]]
            self.weights = [np.random.dirichlet(np.ones(problem.reward_dimension), size=1)[0] for i in xrange(n_vectors)]
            # agents first construction (first weight will be ignored
        else:
            self.weights = weights
        self.r_learning = MORLRLearningAgent(problem, epsilon, alfa, beta, [0.1, ]*problem.reward_dimension)
        self.policies = dict()
        self.rewards = dict()
        self.rhos = dict()
        self.Rs = dict()
        self.weighted_list = dict()
        # storage
        self.converged = False
        self.interactions = interactions
        self.delta = delta
        self.max_per_interaction = max_per_interaction
        self.converging_criterium = converging_criterium
        self.interactions_per_weight = []
        self.stored = dict()
        self.old_rho = np.zeros(self.problem.reward_dimension)
        self.interaction_rhos = []

    def get_learned_action(self, state):
        return self.r_learning.get_learned_action(state)

    def weight_training(self):
        """
        takes n vectors and trains the agent till his weighted average reward converges
        :return:
        """
        # ------PROGRESSBAR START/ LOGGING -----------#
        log.info('Playing  %i interactions on %i vectors...', self.interactions, len(self.weights))
        pbar = pgbar.ProgressBar(widgets=['Weight vector: ', pgbar.SimpleProgress('/'), ' (', pgbar.Percentage(), ') ',
                                 pgbar.Bar(), ' ', pgbar.ETA()], maxval=len(self.weights))
        pbar.start()
        # every weight vector will be used
        for i in xrange(len(self.weights)):
            # put it into the agent
            self.train_one_weight(self.weights[i])
            # plot the evolution of rhos
            #self.plot_interaction_rhos(self.weights[i])
            # evaluate and store policy if good. a True is stored in self.stored[i] if the policy was good enough
            self.stored[i] = self.evaluate_new_policy(self.old_rho, i)
            pbar.update(i)

        self.plot_interactions_per_weight()
        return True

    def plot_interactions_per_weight(self):
        ###################################################################
        #       PLOT (Curve for Learning Process and Policy Plot)         #
        ###################################################################
        fig = plt.figure()
        x = np.arange(len(self.interactions_per_weight))
        plt.plot(x, self.interactions_per_weight, label="interactios per weight")
        for i in range(len(self.stored)):
            if self.stored[i]:
                plt.axvline(i, color='r', linestyle='--')
        plt.axis([0, 1.1*len(self.interactions_per_weight), 0, 1.1*max(self.interactions_per_weight)])
        plt.xlabel("weight count")
        plt.ylabel("count of interactions ")
        plt.draw()

    def look_for_opt(self, weight):
        weighted = [np.dot(weight, self.rhos[u]) for u in self.rhos.iterkeys()]
        max_weighted = max(weighted)
        index_max = weighted.index(max_weighted)
        piopt = self.rhos.keys()[index_max]
        return piopt, weighted

    def evaluate_new_policy(self, old_rho, i):
        """
        this function takes the learned policy and compares the weighted average reward with the same policy
        before learning, if it is better, it stores the new agents params into a pool of "good policies"
        :param old_rho: weighted average reward of policy i before the learning process
        :param i: policy number
        :return:
        """
        policy = PolicyFromAgent(self.problem, self.r_learning, mode='greedy')
        print np.dot(self.weights[i], self.r_learning._rho) - np.dot(self.weights[i], old_rho)
        if np.abs(np.dot(self.weights[i], self.r_learning._rho) - np.dot(self.weights[i], old_rho)) > self.delta:
            # store it
            self.policies[i] = policy
            # and all that other stuff we need later
            self.rewards[i] = self.r_learning._reward
            self.rhos[i] = self.r_learning._rho
            self.Rs[i] = self.r_learning._R
            return True
        else:
            return False

    def plot_interaction_rhos(self, weight):
        interaction_rhos_plot = [np.dot(weight, self.interaction_rhos[r]) for r in xrange(len(self.interaction_rhos))]
        plt.figure()
        plt.axis([0, 1.1*len(interaction_rhos_plot), -1.1*np.abs(min(interaction_rhos_plot)), 1.1*max(interaction_rhos_plot)])
        x = np.arange(len(interaction_rhos_plot))
        plt.plot(x, interaction_rhos_plot, label=str(weight)+' converged: '+str(self.converged))
        plt.xlabel("interactions at this weight")
        plt.ylabel("weighted average reward")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
        self.converged = False

    def train_one_weight(self, weight):
        if len(weight) != self.problem.reward_dimension:
            log.info("could not train this weight, wrong dimension")
            return
        else:
            self.r_learning.w = weight
            # if there are any stored policies
            if self.policies:
                # look for the best
                piopt, weighted = self.look_for_opt(weight)
                self.weighted_list[piopt] = max(weighted)
                # print(weighted[weighted.index(max(weighted))])
                # put its parameters back into the agent
                self.r_learning._rho = self.rhos[piopt]
                self.r_learning._reward = self.rewards[piopt]
                self.r_learning._R = self.Rs[piopt]
            # extract old rho vector
            self.old_rho = self.r_learning._rho
            self.interaction_rhos = []
            # play for interactions times:
            for t in xrange(self.interactions):

                # only for a maximum of epsiodes(without terminating problem)
                for actions in xrange(self.max_per_interaction):
                    # get state of the problem
                    last_state = self.problem.state
                    # take next best action
                    action = self.r_learning.decide(0, self.problem.state)
                    # execute that action
                    payout = self.problem.play(action)
                    # obtain new state
                    new_state = self.problem.state
                    # learn from that action
                    self.r_learning.learn(0, last_state, action, payout, new_state)
                    if self.problem.terminal_state:
                        break
                self.interaction_rhos.append(self.r_learning._rho)
                # check after some interactions if we have converged
                if t > self.converging_criterium:
                    # pick the last phis

                    last_twenty = self.interaction_rhos[t-self.converging_criterium-1:t-1]
                    # pick the last one
                    last_one = self.interaction_rhos[t-1]
                    # create a list that compares all of the twenty with the last
                    compare = np.array([(last_twenty[l] == last_one).all() for l in range(self.converging_criterium)])
                    # if all are same, the algorithm seems to converge
                    if compare.all():
                        self.converged = True
                        break
            # store the count of interactions to show convergence acceleration
            self.interactions_per_weight.append(t)

