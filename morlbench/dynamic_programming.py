#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 16, 2016

@author: Dominik Meyer <meyerd@mytum.de>

    Copyright (C) 2016  Dominik Meyer

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
    def _scalar_reward_solve(self, n_states, gamma, P, R, pi, P_pi):
        """
        Solve the MDP for scalar reward (also used by the MORL classes)

        :param n_states: Number of states in the MPD.
        :param gamma: Discount factor.
        :param P: State transition probabilities.
        :param R: Reward vector.
        :param pi: Policy probability matrix.
        :param P_pi: State transition probability under policy pi.
        :return: Value function as a vector with length n_states.
        """
        return np.dot(npla.inv((np.eye(n_states, n_states) - gamma * P_pi)), R)

    def solve(self):
        """
        Solve the MDP and return the value function, using
        the matrix inversion method.

        :return: Value function as an array.
        """
        n_states, gamma, P, R, pi, P_pi = self._prepare_variables()

        V = self._scalar_reward_solve(n_states, gamma, P, R, pi, P_pi)
        return V


class DynamicProgrammingPolicyEvaluation(DynamicProgramming):
    def _scalar_reward_solve(self, n_states, gamma, P, R, pi, P_pi, max_iterations=10000, vector_implementation=True):
        """
        Solve the MDP for scalar reward (also used by the MORL classes)

        :param n_states: Number of states in the MPD.
        :param gamma: Discount factor.
        :param P: State transition probabilities.
        :param R: Reward vector.
        :param pi: Policy probability matrix.
        :param P_pi: State transition probability under policy pi.
        :param max_iterations: Maximum number of PE iterations
        :return: Value function as a vector with length n_states.
        """

        V = np.zeros(n_states)

        if vector_implementation:
            for i in xrange(max_iterations):
                V_n = R + gamma * np.dot(P_pi, V)
                if npla.norm(V - V_n) < 1e-22:
                    V = V_n
                    log.debug("Policy evaluation converged after %i iterations" % (i))
                    break
                V = V_n
        else:
            delta = float('inf')
            i = 0
            while delta > 1e-2:
                delta = 0
                for s in xrange(n_states):
                    tmp = V[s]
                    a = np.argmax(pi[s, :])
                    V[s] = sum(P[s, a, k] * (R[k] + gamma * V[k]) for k in range(n_states))
                    delta = max(delta, abs(tmp - V[s]))
                    i += 1
                if i > max_iterations:
                    log.warn("Policy evaluation truncated after max_iterations delta: %f" % delta)
                    break
            log.debug("Policy evaluation converged after %i iterations" % (i))

        return V

    def solve(self, max_iterations=10000):
        """
        Solve the MDP using the policy evaluation method.

        :param max_iterations: Max. number of PE iterations (default: 10000)
        :return: Value function as an array.
        """
        n_states, gamma, P, R, pi, P_pi = self._prepare_variables()

        V = self._scalar_reward_solve(n_states, gamma, P, R, pi, P_pi, max_iterations)
        return V


class MORLDynamicProgrammingAbstractSolver(DynamicProgramming):
    def _morl_solve(self, *args, **kwargs):
        """
        Solve the MORL MDP which has a vectorial reward. That
        means the reward matrix R is expected to be in
        (n_states x reward_dimension)

        :return: "value matrix" all the value funcions for the
            individual reward dimensions horizontally stacked
            together.
        """
        n_states, gamma, P, R, pi, P_pi = self._prepare_variables()

        reward_dimensions = R.shape[1]
        V = np.zeros((n_states, reward_dimensions))

        for i in xrange(reward_dimensions):
            V[:, i] = self._scalar_reward_solve(n_states, gamma, P, R[:, i], pi, P_pi, *args, **kwargs)
        return V


class MORLDynamicProgrammingInverse(DynamicProgrammingInverse, MORLDynamicProgrammingAbstractSolver):
    """
    MORL MDP solver using matrix inversion.
    """
    def solve(self, *args, **kwargs):
        return super(MORLDynamicProgrammingInverse, self)._morl_solve(*args, **kwargs)


class MORLDynamicProgrammingPolicyEvaluation(DynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingAbstractSolver):
    """
    MORL MDP solver using Policy Evaluation.
    """
    def solve(self, *args, **kwargs):
        return super(MORLDynamicProgrammingPolicyEvaluation, self)._morl_solve(*args, **kwargs)
