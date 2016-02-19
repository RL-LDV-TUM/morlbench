#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import virtualFunction, SaveableObject

import numpy as np
import random
import logging as log

try:
    # import neurolab only if it exists in case it is not used and not installed
    # such that the other agents still work
    import neurolab as nl
except ImportError, e:
    log.warn("Neurolab not installed: %s" % (str(e)))

# log.basicConfig(level=log.DEBUG)


class MorlAgent(SaveableObject):
    """
    A agent that should interface with a MORL problem.
    """

    def __init__(self, morl_problem, **kwargs):
        """
        Initialize the Agent with the MORL problem
        problem, it will be faced with.

        Parameters
        ----------
        morl_problem: The already initialized and
            correctly parametrized problem.
        """

        super(MorlAgent, self).__init__([])

        self._morl_problem = morl_problem

    def __str__(self):
        return self.__class__.__name__

    def learn(self, t, action, reward, state):
        """
        Learn on the last interaction specified by the
        action and the reward received.

        :param t: Interaction cycle we are currently in
        :param action: last interaction action
        :param reward: received reward vector
        :param state: next state transited to
        :return:
        """
        # virtualFunction()

    def decide(self, t, state):
        """
        Decide which action to take in interaction
        cycle t.

        Parameters
        ----------
        :param t: Interaction cycle we are currently in
        :param state: state we are in

        Returns
        -------
        action: The action to do next
        """
        virtualFunction()

    def learn(self, t, action, reward, state):
        """
        Learn from the last interaction, if we have
        a dynamically learning agent.

        Parameters
        ----------
        :param t: int Interaction cycle.
        :param action: last interaction action
        :param reward: received reward vector
        :param state: next state transited to
        """
        virtualFunction()


class TDMorlAgent(MorlAgent):
    """
    A MORL agent, that uses TD for Policy Evaluation.
    """

    def __init__(self, problem, scalarization_weights, policy, alpha=0.3, **kwargs):
        """
        Initialize the TD Policy Evaluation learner for MORL.
        Scalarization weights have to be given.

        :param problem: MORL problem.
        :param scalarization_weights: Reward scalarization weights.
        :param policy: A static policy, that will be evaluated.
        :param alpha: Learning rate.
        """
        super(TDMorlAgent, self).__init__(problem, **kwargs)

        self._policy = policy
        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._gamma = self._morl_problem.gamma

        self._V = np.zeros(self._morl_problem.n_states)
        self._last_state = 0
        self._last_action = random.randint(0,problem.n_actions-1)
        self._last_reward =  np.zeros_like(self._scalarization_weights)

    def learn(self, t, action, reward, state):
        self._learn(0, self._last_state, self._last_action,
                    self._last_reward, action, reward, state)
        self._last_action = action
        self._last_reward = reward
        self._last_state = state

    def _learn(self, t, last_state, last_action, last_reward, action, reward, state):
        scalar_reward = np.dot(self._scalarization_weights.T, reward)
        self._V[last_state]  += self._alpha * (scalar_reward + self._gamma * self._V[state] - self._V[last_state])

        log.debug(' V: %s' % (str(self._V)))

    def decide(self, t, state):
        return self._policy.decide(state)


class SARSAMorlAgent(MorlAgent):
    """
    A MORL agent, that uses RL.
    """

    def __init__(self, problem, scalarization_weights, alpha=0.3, epsilon=1.0, **kwargs):
        """
        Initialize the Reinforcement Learning MORL
        Agent with the problem description and alpha,
        the learning rate.

        Parameters
        ----------
        :param problem: A MORL problem
        :param scalarization_weights: a weight vector to scalarize the morl reward.
        :param alpha: real, the learning rate in each
            SARSA update step
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(SARSAMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._gamma = self._morl_problem.gamma
        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._last_state = 0
        self._last_action = random.randint(0,problem.n_actions-1)
        self._last_reward = np.zeros_like(self._scalarization_weights)

    def learn(self, t, action, reward, state):
        self._learn(0, self._last_state, self._last_action,
                    self._last_reward, action, reward, state)
        self._last_action = action
        self._last_reward = reward
        self._last_state = state

    def _learn(self, t, last_state, last_action, last_reward, action, reward, state):
        scalar_reward = np.dot(self._scalarization_weights.T, reward)
        self._Q[last_state, last_action] += self._alpha * \
                                            (scalar_reward + self._gamma * self._Q[state, action] - self._Q[
                                                last_state, last_action])
        log.debug(' Q: %s' % (str(self._Q)))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            action = random.choice(np.where(self._Q[state, :] == max(self._Q[state, :]))[0])
            #action = self._Q[state, :].argmax()
            log.debug('  took greedy action %i' % (action))
            return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        log.debug('   took random action %i' % (action))
        return action

    def get_learned_action(self, state):
        return self._Q[state, :].argmax()


class QMorlAgent(MorlAgent):
    """
    A MORL agent, that uses Q learning.
    """

    def __init__(self, problem, scalarization_weights, alpha=0.3, epsilon=1.0, **kwargs):
        """
        Initialize the Reinforcement Learning MORL
        Agent with the problem description and alpha,
        the learning rate.

        Parameters
        ----------
        :param problem: A MORL problem
        :param scalarization_weights: a weight vector to scalarize the morl reward.
        :param alpha: real, the learning rate in each
            Q update step
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(QMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._gamma = self._morl_problem.gamma
        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros(
                (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        # self._Q = np.ones(
        #         (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        self._last_state = 0
        self._last_action = random.randint(0,problem.n_actions-1)
        self._last_reward = np.zeros_like(self._scalarization_weights)

    def learn(self, t, action, reward, state):
        self._learn(0, self._last_state, self._last_action,
                    self._last_reward, action, reward, state)
        self._last_action = action
        self._last_reward = reward
        self._last_state = state

    def _learn(self, t, last_state, last_action, last_reward, action, reward, state):
        """
        Updating the Q-table according to Suttons Q-learning update for multiple
        objectives
        :param t: unused
        :param last_state:
        :param last_action:
        :param last_reward:
        :param action:
        :param reward:
        :param state:
        :return:
        """

        # scalar_reward = np.dot(self._scalarization_weights.T, reward)

        self._Q[last_state, last_action] += self._alpha * \
                                  (reward + self._gamma * np.amax(self._Q[state, :], axis=0) - self._Q[
                                      last_state, last_action])

        log.debug(' Q: %s' % (str(self._Q[state, :, :])))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            weighted_q = np.dot(self._Q[state, :], self._scalarization_weights)
            action = random.choice(np.where(weighted_q == max(weighted_q))[0])

            log.debug('  took greedy action %i' % action)
            return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        log.debug('   took random action %i' % action)
        return action

    def get_learned_action(self, state):
        return np.dot(self._Q[state, :], self._scalarization_weights).argmax()


class DeterministicAgent(MorlAgent):

    def __init__(self, morl_problem, **kwargs):
        super(DeterministicAgent, self).__init__(morl_problem, **kwargs)

        self._transition_dict = {0:   2,
                                 1:   1,
                                 11:  2,
                                 12:  1,
                                 22:  2,
                                 23:  1,
                                 33:  2,
                                 34:  2,
                                 35:  2,
                                 36:  1,
                                 46:  1,
                                 56:  1,
                                 66:  1
                                 }

    def decide(self, t, state):
        if state in self._transition_dict:
            return self._transition_dict[state]
        else:
            return random.randint(0, self._morl_problem.n_actions-1)

    def learn(self, t, action, reward, state):
        pass

class NFQAgent(MorlAgent):

    def __init__(self, morl_problem, scalarization_weights, gamma, epsilon, **kwargs):
        super(NFQAgent, self).__init__(morl_problem, **kwargs)

        self._gamma = gamma
        self._epsilon = epsilon
        self._scalarization_weights = scalarization_weights

        self._transistion_history = []  # full transition history (s,a,a')
        self._train_history = []  # input history for NN (s,a)
        self._goal_hist = []  # goal history
        self._last_state = random.randint(0,morl_problem.n_actions-1)

        # Create network with 2 layers and random initialized
        self._net = nl.net.newff([[0, self._morl_problem.n_states], [0, 3]], [20, 20, 1])

    def learn(self, t, action, reward, state):
        # Generate training set
        self._transistion_history.append([state, action, self._last_state])
        self._train_history.append([state, action])

        Q_vals = []
        for i in xrange(self._morl_problem.n_actions - 1):
            # Simulate network
            Q_vals.append(self._net.sim(np.asarray([[state, i]])))

        # cost function (minimum time controller)
        if self._morl_problem.terminal_state:
            costs = 0
            self._goal_hist.append(costs + self._gamma * 1/reward[0])
        else:
            costs = -1
            self._goal_hist.append(costs + self._gamma * np.array(Q_vals).min())

        inp = np.asarray(self._train_history)
        tar = np.asarray(self._goal_hist)
        tar = tar.reshape(len(tar), 1)

        # Train network
        # error = self._net.train.train_rprop(input, target, epochs=500, show=100, goal=0.02)
        nl.train.train_rprop(self._net, inp, tar, epochs=500, show=100, goal=0.02)


    def decide(self, t, state):
        Q_vals = []
        for i in xrange(self._morl_problem.n_actions):
            # Simulate network
            Q_vals.append(self._net.sim(np.asarray([[state, i]])))

        action = random.choice(np.where(np.array(Q_vals) == min(np.array(Q_vals)))[0])

        # if random.random() < self._epsilon:
        #     weighted_q = np.dot(self._Q[state, :], self._scalarization_weights)
        #     action = random.choice(np.where(weighted_q == max(weighted_q))[0])
        #
        #     log.debug('  took greedy action %i' % action)
        #     return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        log.debug('   took random action %i' % action)
        return action

        log.debug('Decided for action %i in state %i.', action, state)

        return action
