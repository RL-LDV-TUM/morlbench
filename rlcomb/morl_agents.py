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


class SARSAMorlAgent(MorlAgent):
    """
    A MORL agent, that uses RL.
    """

    def __init__(self, problem, scalarization_weights, alpha=0.3, gamma=1.0, epsilon=1.0, **kwargs):
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
        :param gamma: real, [0, 1) RL discount factor
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(SARSAMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._last_state = 0
        self._last_action = 0
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
                                                state, last_action])
        log.debug(' Q: %s' % (str(self._Q)))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            action = self._Q[state, :].argmax()
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

    def __init__(self, problem, scalarization_weights, alpha=0.3, gamma=1.0, epsilon=1.0, **kwargs):
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
        :param gamma: real, [0, 1) RL discount factor
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(QMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros(
                (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        self._last_state = 0
        self._last_action = 0
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

        self._Q[state, action] += self._alpha * \
                                  (reward + self._gamma * np.amax(self._Q[state, :], axis=0) - self._Q[
                                      state, last_action])

        log.debug(' Q: %s' % (str(self._Q[state, :, :])))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            action = np.dot(self._Q[state, :], self._scalarization_weights).argmax()

            log.debug('  took greedy action %i' % (action))
            return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        log.debug('   took random action %i' % (action))
        return action

    def get_learned_action(self, state):
        return np.dot(self._Q[state, :], self._scalarization_weights).argmax()
