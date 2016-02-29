#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 16, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import SaveableObject
from probability_helpers import assurePolicyMatrix, sampleFromDiscreteDistribution

import numpy as np
import random
import logging as log


class PolicyDeepsea(SaveableObject):
    """
    This is the base class for all Deepsea problem policies.
    """

    def __init__(self, problem):
        """
        Initialize the Deepsea problem base policy

        Parameters
        ----------
        :param problem: Initialized Deepsea problem.
        """

        super(PolicyDeepsea, self).__init__([])
        self._problem = problem
        self._pi = None

    def __str__(self):
        return self.__class__.__name__

    def get_pi(self):
        """
        Return the policy matrix :math:`\Pi` which consists of
        m rows, where m is the number of states
        of the problem and n columns, where n is the number of actions.

        :return: :math:`\Pi \in \mathbb{R}^{m \times n}`
        """
        return self._pi

    def get_pi_a(self, a):
        """
        Return the a-th column of the policy matrix, this means
        the probability distribution for a specific action given a
        specific state. m is the number of states.

        :param a: int action
        :return: :math:`\Pi^{a} \in \mathbb{R}^{m \times 1}`
        """
        return self._pi[:, a]

    def decide(self, state):
        """
        Decide which action to take in state.
        :param state: Deepsea state tuple.
        :return: int action.
        """
        return sampleFromDiscreteDistribution(1, self._pi[state, :])

    def get_optimal_action(self, state):
        """

        :param state:
        :return:
        """
        # return the action with the max probability, break ties
        # in a random manner
        return random.choice(np.where(self._pi[state, :] == max(self._pi[state, :]))[0])


class PolicyDeepseaRandom(PolicyDeepsea):
    """
    A random policy for the Deepsea MORL problem.
    """
    def __init__(self, problem):
        super(PolicyDeepseaRandom, self).__init__(problem)

        # randomly initialize policy
        # TODO: only put ones to valid actions
        self._pi = np.ones((self._problem.n_states, self._problem.n_actions))
        # normalize
        self._pi /= self._pi.sum(axis=1)[:, np.newaxis]
        assurePolicyMatrix(self._pi)


class PolicyDeepseaDeterministicExample01(PolicyDeepsea):
    """
    A deterministic example policy for the deepsea scenario.
    """
    _transition_dict = {0:   2,
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

    def __init__(self, problem):
        super(PolicyDeepseaDeterministicExample01, self).__init__(problem)

        self._pi = np.zeros((self._problem.n_states, self._problem.n_actions))

        # fill pi for all states with 1/n_actions except for the ones in
        # transition_dict
        for i in xrange(self._problem.n_states):
            if i in self._transition_dict:
                self._pi[i, self._transition_dict[i]] = 1.0
            else:
                self._pi[i, :] = 1.0 / self._problem.n_actions


class PolicyDeepseaExpert(PolicyDeepsea):
    """
    Human expert policy for the deepsea scenario
    """
                  # state:   up,   do,   ri,   le,
    _T1 = {     0:  (0.00, 0.00, 1.00, 0.00),
                1:  (0.00, 0.10, 0.90, 0.00),
                2:  (0.00, 0.22, 0.78, 0.00),
                3:  (0.00, 0.00, 1.00, 0.00),
                4:  (0.00, 0.00, 1.00, 0.00),
                5:  (0.00, 0.14, 0.86, 0.00),
                6:  (0.00, 0.00, 1.00, 0.00),
                7:  (0.00, 0.50, 0.50, 0.00),
                8:  (0.00, 0.33, 0.67, 0.00),
                9:  (0.00, 1.00, 0.00, 0.00),
                11: (0.00, 0.00, 1.00, 0.00),
                12: (0.00, 0.33, 0.67, 0.00),
                13: (0.00, 0.50, 0.50, 0.00),
                14: (0.00, 1.00, 0.00, 0.00),
                15: (0.00, 1.00, 0.00, 0.00),
                17: (0.00, 1.00, 0.00, 0.00),
                18: (0.00, 1.00, 0.00, 0.00),
                19: (0.00, 1.00, 0.00, 0.00),
                22: (0.00, 0.00, 1.00, 0.00),
                23: (0.00, 0.50, 0.50, 0.00),
                24: (0.00, 0.00, 1.00, 0.00),
                25: (0.00, 0.50, 0.50, 0.00),
                26: (0.00, 0.00, 1.00, 0.00),
                27: (0.00, 0.80, 0.20, 0.00),
                28: (0.00, 0.50, 0.50, 0.00),
                29: (0.00, 1.00, 0.00, 0.00),
                33: (0.00, 0.00, 1.00, 0.00),
                34: (1.00, 0.00, 0.00, 0.00),
                35: (0.00, 1.00, 0.00, 0.00),
                37: (0.00, 1.00, 0.00, 0.00),
                38: (0.00, 1.00, 0.00, 0.00),
                39: (0.00, 1.00, 0.00, 0.00),
                47: (0.00, 1.00, 0.00, 0.00),
                48: (0.00, 1.00, 0.00, 0.00),
                49: (0.00, 1.00, 0.00, 0.00),
                57: (0.00, 1.00, 0.00, 0.00),
                58: (0.00, 1.00, 0.00, 0.00),
                59: (0.00, 1.00, 0.00, 0.00),
                67: (0.00, 1.00, 0.00, 0.00),
                68: (0.00, 1.00, 0.00, 0.00),
                69: (0.00, 1.00, 0.00, 0.00),
                78: (0.00, 1.00, 0.00, 0.00),
                79: (0.00, 1.00, 0.00, 0.00),
                88: (0.00, 1.00, 0.00, 0.00),
                89: (0.00, 1.00, 0.00, 0.00),
                99: (0.00, 1.00, 0.00, 0.00)
                }

    def __init__(self, problem, task='T1'):
        super(PolicyDeepseaExpert, self).__init__(problem)

        self._pi = np.zeros((self._problem.n_states, self._problem.n_actions))

        if task == 'T1':
            pi_dict = self._T1
        else:
            raise ValueError('Given Task Policy does not exist')

        # fill pi for all states with 1/n_actions except for the ones in
        # transition_dict
        for i, vals in pi_dict.iteritems():
            for action, val in enumerate(vals):
                self._pi[i, action] = val

        # TODO: implement the terminal state and the "do-nothing-action"


class PolicyDeepseaFromAgent(PolicyDeepsea):
    """
    Derive a greedy policy from a trained agent.
    """
    def __init__(self, problem, agent, mode='gibbs'):
        super(PolicyDeepseaFromAgent, self).__init__(problem)

        self._agent = agent
        self._pi = np.zeros((self._problem.n_states, self._problem.n_actions))

        for i in xrange(self._problem.n_states):
            if mode == 'gibbs':
                # gibbs
                a_dist = agent.get_learned_action_distribution(i)
                self._pi[i, :] = a_dist
            else:
                # greedy
                a = agent.get_learned_action(i)
                self._pi[i, a] = 1.0

