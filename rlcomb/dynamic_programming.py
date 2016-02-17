#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 16, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import SaveableObject, virtualFunction

import numpy as np
import numpy.linalg as npla
import logging as log


class DynamicProgramming(SaveableObject):
    """
    Dynamic Programming base class to solve MDPs, which
    can provide a state transition matrix, a reward vector
    and policies, which can provide policy probability
    matrices.
    """

    def __init__(self, problem, policy):
        """
        Initialize the Dynamic Programming solver.

        :param problem: A MDP problem.
        :param policy: A matching policy for the problem.
        """
        super(DynamicProgramming, self).__init__([])

        self._problem = problem
        self._policy = policy

    def _prepare_variables(self):
        n_states = self._problem.n_states
        gamma = self._problem.gamma
        P = self._problem.P
        R = self._problem.R
        pi = self._policy.get_pi()

        # calculate state transition probability matrix together with policy
        P_pi = P * pi[:, :, np.newaxis]
        P_pi = P_pi.sum(axis=1)
        P_pi /= P_pi.sum(axis=1)[:, np.newaxis]
        return (n_states, gamma, P, R, pi, P_pi)

    def solve(self):
        """
        Solve the MDP.
        :return: Value function as an array.
        """
        virtualFunction()


class DynamicProgrammingInverse(DynamicProgramming):
    def solve(self):
        """
        Solve the MDP and return the value function, using
        the matrix inversion method.

        :return: Value function as an array.
        """
        n_states, gamma, P, R, pi, P_pi = self._prepare_variables()

        V = np.dot(npla.inv((np.eye(n_states, n_states) - gamma * P_pi)), R)
        return V


class DynamicProgrammingValueIteration(DynamicProgramming):
    def solve(self, max_iterations=10000):
        """
        Solve the MDP using the policy evaluation method.

        :param max_iterations: Max. number of PE iterations (default: 10000)
        :return: Value function as an array.
        """
        n_states, gamma, P, R, pi, P_pi = self._prepare_variables()

        V = np.zeros(n_states)
        for i in xrange(max_iterations):
            V_n = R + gamma * np.dot(P_pi, V)
            if npla.norm(V - V_n) < 1e-20:
                V = V_n
                log.debug("Value iteration converged after %i iterations" % (i))
                break
            V = V_n
        return V
