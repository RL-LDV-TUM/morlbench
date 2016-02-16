"""
Created on Feb 16, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import SaveableObject
from probability_helpers import assurePolicyMatrix, sampleFromDiscreteDistribution

import numpy as np
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
        return sampleFromDiscreteDistribution(1, self._pi[state, :])[0]


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
