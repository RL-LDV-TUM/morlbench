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
        # Discount Factor is part of the problem, but is used in most algorithms
        self._gamma = morl_problem.gamma

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

    def learn(self, t, last_state, action, reward, state):
        """
        Learn from the last interaction, if we have
        a dynamically learning agent.

        Parameters
        ----------
        :param t: int Interaction cycle.
        :param last_state: Last state where we came from
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

        self._V = np.zeros(self._morl_problem.n_states)

    def learn(self, t, last_state, action, reward, state):
        scalar_reward = np.dot(self._scalarization_weights.T, reward)
        self._V[last_state] += self._alpha * (scalar_reward + self._gamma * self._V[state] - self._V[last_state])

        log.debug(' V: %s' % (str(self._V[0:110].reshape((11, 10)))))

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

        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._last_action = random.randint(0,problem.n_actions-1)

    def learn(self, t, last_state, action, reward, state):
        self._learn(0, last_state, self._last_action, action, reward, state)
        self._last_action = action

    def _learn(self, t, last_state, last_action, action, reward, state):
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
        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros(
                (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        # self._Q = np.ones(
        #         (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        self._last_action = random.randint(0,problem.n_actions-1)

    def learn(self, t, last_state, action, reward, state):
        self._learn(0, self._last_state, self._last_action,
                    self._last_reward, reward, state)
        self._last_action = action

    def _learn(self, t, last_state, last_action, reward, state):
        """
        Updating the Q-table according to Suttons Q-learning update for multiple
        objectives
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


class FixedPolicyAgent(MorlAgent):
    """
    An agent, that follows a fixed policy, defined as a morl policy object.
    No learning implemented.
    """
    def __init__(self, morl_problem, policy, **kwargs):
        super(FixedPolicyAgent, self).__init__(morl_problem, **kwargs)

        self._policy = policy

    def learn(self, t, last_state, action, reward, state):
        pass

    def decide(self, t, state):
        return self._policy.decide(state)


class NFQAgent(MorlAgent):
    """
    Implements neural fitted-Q iteration

    Can be used with scenarios where the reward vector contains only
    values between -1 and 1.

    TODO: Currently only morl problems with a cartesian coordinate
    system as state can be used.
    """
    def __init__(self, morl_problem, scalarization_weights, gamma, epsilon, **kwargs):
        super(NFQAgent, self).__init__(morl_problem, **kwargs)

        self._gamma = gamma
        self._epsilon = epsilon
        self._scalarization_weights = scalarization_weights

        self._transistion_history = []  # full transition history (s,a,a')
        self._train_history = []  # input history for NN (s,a)
        self._goal_hist = []  # goal history
        self._last_state = 0
        self._last_action = random.randint(0, morl_problem.n_actions-1)
        self._last_reward = np.zeros_like(self._scalarization_weights)

        # Create network with 2 layers and random initialized
        # self._net = nl.net.newff([[0, self._morl_problem.n_states], [0, 3]], [20, 20, 1])
        self._net = nl.net.newff([[0, self._morl_problem.scene_y_dim],
                                  [0, self._morl_problem.scene_x_dim],
                                  [0, self._morl_problem.n_actions]],
                                 [20, 20, len(self._scalarization_weights)])

    def learn(self, t, action, reward, state):
        self._learn(0, self._last_state, self._last_action,
                    self._last_reward, action, reward, state)
        # state transition and update
        self._last_state = state
        self._last_action = action
        self._last_reward = reward

    def _learn(self, t, last_state, last_action, last_reward, action, reward, state):
        self._transistion_history.append([last_state, action, state])

        # Generate training set
        #self._train_history.append([last_state, last_action])
        self._train_history.append(np.hstack([np.array(self._morl_problem._get_position(last_state)), last_action]))


        Q_vals = []
        for i in xrange(self._morl_problem.n_actions):
            # Simulate network
            tmp = np.hstack([np.array(self._morl_problem._get_position(state)), i])
            Q_vals.append(self._net.sim(np.array([tmp])))



        tar_tmp = (reward + self._gamma * np.amax(Q_vals, axis=0))
        self._goal_hist.append(tar_tmp)


        # cost function (minimum time controller)
        # costs = 0.01
        # if self._morl_problem.terminal_state:
        #     costs += self._gamma * (1.0/reward[0])
        #     self._goal_hist.append(costs)
        #     log.debug('Terminal cost value in state %i is %f', state, costs)
        # else:
        #     Q_vals = []
        #     for i in xrange(self._morl_problem.n_actions):
        #         # Simulate network
        #         # Q_vals.append(self._net.sim(np.asarray([[state, i]])))
        #         tmp = np.hstack([np.array(self._morl_problem._get_position(state)), i])
        #         Q_vals.append(self._net.sim(np.array([tmp])))
        #
        #     costs += self._gamma * np.array(Q_vals).min()
        #     self._goal_hist.append(costs)
        #     log.debug('State cost value in state %i is %f', state, costs)

        inp = np.array(self._train_history)
        tar = np.array(self._goal_hist)
        #tar = tar.reshape(len(tar), 1)
        tar = tar.reshape(len(tar), 2) # TODO: fix me for arbitrary reward dimensions

        # Train network
        # error = self._net.train.train_rprop(input, target, epochs=500, show=100, goal=0.02)
        nl.train.train_rprop(self._net, inp, tar, epochs=500, show=0, goal=0.02)

        # Reset histories after termination of one episode
        if self._morl_problem.terminal_state:
            self._train_history = []  # input history for NN (s,a)
            self._goal_hist = []  # goal history


    def decide(self, t, state):
        # epsilon greedy
        if random.random() < self._epsilon:
            Q_vals = []
            for i in xrange(self._morl_problem.n_actions):
                # Simulate network
                # Q_vals.append(self._net.sim(np.asarray([[state, i]])))
                tmp = np.hstack([np.array(self._morl_problem._get_position(state)), i])
                Q_vals.append(self._net.sim(np.array([tmp])))

            weighted_q = np.dot(Q_vals, self._scalarization_weights)
            action = random.choice(np.where(weighted_q == max(weighted_q))[0])
            # action = random.choice(np.where(np.array(Q_vals) == min(np.array(Q_vals)))[0])
        else:
            action = random.randint(0, self._morl_problem.n_actions - 1)

        log.debug('Decided for action %i in state %i.', action, state)

        return action
