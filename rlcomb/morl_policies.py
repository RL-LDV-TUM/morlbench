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


class Policy(SaveableObject):
    """
    This is the base class for all problem policies.
    """

    def __init__(self, problem):
        """
        Initialize the problem base policy

        Parameters
        ----------
        :param problem: Initialized Deepsea problem.
        """

        super(Policy, self).__init__([])
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


class PolicyDeepseaRandom(Policy):
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


class PolicyDeepseaDeterministic(Policy):
    """
    A deterministic example policy for the deepsea scenario.
    """
    # to the 24
    _P1 = {0:   2,
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
    # greedy to treasure 1
    _P2 = {0:   1
           }
    # flat to the right, then down to 124
    _P3 = {0:   2,
           1:   2,
           2:   2,
           3:   2,
           4:   2,
           5:   2,
           6:   2,
           7:   2,
           8:   2,
           9:   1,
           19:   1,
           29:   1,
           39:   1,
           49:   1,
           59:   1,
           69:   1,
           79:   1,
           89:   1,
           99:   1,
           }
    # first flat then down to 3
    _P4 = {0:   2,
           1:   2,
           2:   1,
           12:  1,
           22:  1
           }
    # diagonal to 8
    _P5 = {0:   2,
           1:   1,
           11:  2,
           12:  1,
           22:  2,
           23:  1,
           33:  2,
           34:  1
           }

    def __init__(self, problem, policy='P1'):
        super(PolicyDeepseaDeterministic, self).__init__(problem)

        self._pi = np.zeros((self._problem.n_states, self._problem.n_actions))

        if policy == 'P1':
            transition_dict = self._P1
        elif policy =='P2':
            transition_dict = self._P2
        elif policy =='P3':
            transition_dict = self._P3
        elif policy =='P4':
            transition_dict = self._P4
        elif policy =='P5':
            transition_dict = self._P5
        else:
            raise ValueError('Given Policy does not exist')

        # fill pi for all states with 1/n_actions except for the ones in
        # transition_dict
        for i in xrange(self._problem.n_states):
            if i in transition_dict:
                self._pi[i, transition_dict[i]] = 1.0
            else:
                # self._pi[i, :] = 1.0 / self._problem.n_actions
                self._pi[i, -1] = 1.0


class PolicyDeepseaExpert(Policy):
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

    # T2 -> risky captain
          # state:   up,   do,   ri,   le,
    _T2 = {     0:  (0.00, 0.00, 1.00, 0.00),
                1:  (0.00, 0.10, 0.90, 0.00),
                2:  (0.00, 0.11, 0.89, 0.00),
                3:  (0.00, 0.00, 1.00, 0.00),
                4:  (0.00, 0.00, 1.00, 0.00),
                5:  (0.00, 0.00, 1.00, 0.00),
                6:  (0.00, 0.00, 1.00, 0.00),
                7:  (0.00, 0.12, 0.88, 0.00),
                8:  (0.00, 0.29, 0.71, 0.00),
                9:  (0.00, 1.00, 0.00, 0.00),
                11: (0.00, 0.00, 1.00, 0.00),
                12: (0.00, 0.50, 0.50, 0.00),
                13: (0.00, 0.00, 1.00, 0.00),
                14: (0.00, 1.00, 0.00, 0.00),
                17: (0.00, 1.00, 0.00, 0.00),
                18: (0.00, 1.00, 0.00, 0.00),
                19: (0.00, 1.00, 0.00, 0.00),
                22: (0.00, 0.00, 1.00, 0.00),
                23: (0.00, 0.00, 1.00, 0.00),
                24: (0.00, 0.00, 1.00, 0.00),
                25: (0.00, 0.00, 1.00, 0.00),
                26: (0.00, 1.00, 0.00, 0.00),
                27: (0.00, 1.00, 0.00, 0.00),
                28: (0.00, 1.00, 0.00, 0.00),
                29: (0.00, 1.00, 0.00, 0.00),
                36: (0.00, 1.00, 0.00, 0.00),
                37: (0.00, 1.00, 0.00, 0.00),
                38: (0.00, 1.00, 0.00, 0.00),
                39: (0.00, 1.00, 0.00, 0.00),
                46: (0.00, 1.00, 0.00, 0.00),
                47: (0.00, 1.00, 0.00, 0.00),
                48: (0.00, 1.00, 0.00, 0.00),
                49: (0.00, 1.00, 0.00, 0.00),
                56: (0.00, 0.50, 0.50, 0.00),
                57: (0.00, 0.50, 0.50, 0.00),
                58: (0.00, 0.67, 0.33, 0.00),
                59: (0.00, 1.00, 0.00, 0.00),
                66: (0.00, 1.00, 0.00, 0.00),
                67: (0.00, 0.50, 0.50, 0.00),
                68: (0.00, 0.67, 0.33, 0.00),
                69: (0.00, 1.00, 0.00, 0.00),
                78: (0.00, 1.00, 0.00, 0.00),
                79: (0.00, 1.00, 0.00, 0.00),
                88: (0.00, 1.00, 0.00, 0.00),
                89: (0.00, 1.00, 0.00, 0.00),
                99: (0.00, 1.00, 0.00, 0.00)
                }

    # T3 -> defensive captain
          # state:   up,   do,   ri,   le,
    _T3 = {     0:  (0.00, 0.00, 1.00, 0.00),
                1:  (0.00, 0.00, 1.00, 0.00),
                2:  (0.00, 0.10, 0.90, 0.00),
                3:  (0.00, 0.11, 0.89, 0.00),
                4:  (0.00, 0.00, 1.00, 0.00),
                5:  (0.00, 0.62, 0.38, 0.00),
                6:  (0.00, 0.00, 1.00, 0.00),
                7:  (0.00, 1.00, 0.00, 0.00),
                12: (0.00, 1.00, 0.00, 0.00),
                13: (0.00, 1.00, 0.00, 0.00),
                15: (0.00, 1.00, 0.00, 0.00),
                17: (0.00, 1.00, 0.00, 0.00),
                22: (0.00, 1.00, 0.00, 0.00),
                23: (0.00, 1.00, 0.00, 0.00),
                25: (0.00, 1.00, 0.00, 0.00),
                27: (0.00, 1.00, 0.00, 0.00),
                33: (0.00, 1.00, 0.00, 0.00),
                35: (0.00, 1.00, 0.00, 0.00),
                37: (0.00, 1.00, 0.00, 0.00),
                47: (0.00, 1.00, 0.00, 0.00),
                57: (0.00, 1.00, 0.00, 0.00),
                67: (0.00, 1.00, 0.00, 0.00)
                }

    def __init__(self, problem, task='T1'):
        super(PolicyDeepseaExpert, self).__init__(problem)

        self._pi = np.zeros((self._problem.n_states, self._problem.n_actions))

        if task == 'T1':
            pi_dict = self._T1
        elif task == 'T2':
            pi_dict = self._T2
        elif task == 'T3':
            pi_dict = self._T3
        else:
            raise ValueError('Given Task Policy does not exist')

        self._pi[:, -1] = 1.0

        # fill pi for all states with 1/n_actions except for the ones in
        # transition_dict
        for i, vals in pi_dict.iteritems():
            for action, val in enumerate(vals):
                self._pi[i, action] = val
            self._pi[i, -1] = 0.0

        # TODO: implement the terminal state!?


class PolicyDeepseaFromAgent(Policy):
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


class PolicyGridworld(Policy):
    """
    Optimal policy for the gridworld is to go diagonal from the upper left
    to the lower right.
    """
    def __init__(self, problem, policy='DIAGONAL'):
        super(PolicyGridworldExample, self).__init__(problem)

        self._pi = np.zeros((self._problem.n_states, self._problem.n_actions))

        if policy == 'DIAGONAL':
            for i in xrange(self._problem.n_states):
                # TODO: this is a private function and should be refactored
                x, y = self._problem._get_position(i)
                if x < y:
                    self._pi[i, 0] = 1.0
                else:
                    self._pi[i, 1] = 1.0
        elif policy == 'DOWN':
            for i in xrange(self._problem.n_states):
                x, y = self._problem._get_position(i)
                # TODO: check if this only works for square worlds or also in general!
                if x + y < self._problem.scene_x_dim:
                    self._pi[i, 1] = 1.0
                else:
                    self._pi[i, 2] = 1.0
        elif policy == 'RIGHT':
            for i in xrange(self._problem.n_states):
                x, y = self._problem._get_position(i)
                if x + y < self._problem.scene_x_dim:
                    self._pi[i, 0] = 1.0
                else:
                    self._pi[i, 3] = 1.0
