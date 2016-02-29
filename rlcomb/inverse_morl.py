#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import SaveableObject, virtualFunction
from dynamic_programming import DynamicProgrammingInverse

import numpy as np
import numpy.linalg as npla
import logging as log
try:
    from cvxopt import solvers, matrix
except ImportError:
    log.warn("cvxopt module not found. Inverse RL functions not available.")


class InverseMORL(SaveableObject):
    """
    Inverse RL for finding scalarization weights in Multi-Objective
    Reinforcement Learning problems.
    """

    def __init__(self, problem, policy):
        """
        Initialize the Inverse MORL solver.

        :param problem: A MDP problem.
        :param policy: The optimal policy for the problem.
        """
        super(InverseMORL, self).__init__([])

        self._problem = problem
        self._policy = policy

    def _prepare_variables(self):
        n_states = self._problem.n_states
        n_actions = self._problem.n_actions
        reward_dimension = self._problem.reward_dimension
        gamma = self._problem.gamma
        P = self._problem.P
        R = self._problem.R
        pi = self._policy.get_pi()

        # calculate state transition probability matrix together with policy
        P_pi = P * pi[:, :, np.newaxis]
        P_pi = P_pi.sum(axis=1)
        P_pi /= P_pi.sum(axis=1)[:, np.newaxis]
        return (n_states, n_actions, reward_dimension, gamma, P, R, pi, P_pi)

    def _prepare_v(self, n_states, n_actions, reward_dimension, P):
        problem, policy = self._problem, self._policy

        dp = DynamicProgrammingInverse(problem, policy)
        V = dp.solve()

        v = np.zeros((n_states, n_actions - 1, reward_dimension))

        for i in xrange(n_states):
            optimal_a = policy.get_optimal_action(i)

            v_i_optimal = np.dot(P[i, optimal_a, :].T, V)

            already_had_optimal = False
            for a in xrange(n_actions):
                if a == optimal_a:
                    already_had_optimal = True
                    continue

                v_i_a = np.dot(P[i, a, :].T, V)

                if already_had_optimal:
                    v[i, a - 1, :] = v_i_optimal - v_i_a
                else:
                    v[i, a, :] = v_i_optimal - v_i_a
        return v

    def _vertical_ones(self, n, m, i):
        """
        Construct a (n x m) matrix with ones on the i-th column
        :return: (n x m) array
        """
        r = np.zeros((n, m))
        r[:, i] = 1.0
        return r

    def solve(self):
        """
        Solve the Inverse RL problem
        :return: Scalarization weights as array.
        """
        n_states, n_actions, reward_dimension, gamma, P, R, pi, P_pi = self._prepare_variables()

        v = self._prepare_v(n_states, n_actions, reward_dimension, P)

        c = np.vstack([np.ones((n_states, 1)), np.zeros((reward_dimension, 1))])
        G = np.vstack([
            np.hstack([np.zeros((reward_dimension, n_states)), np.eye(reward_dimension)]),
            np.hstack([np.zeros((reward_dimension, n_states)), -np.eye(reward_dimension)]),
            np.vstack([
                np.vstack([
                   np.hstack([self._vertical_ones(1, n_states, i), -v[i, j, :].reshape(1, -1)]) for j in xrange(n_actions - 1)
                ]) for i in xrange(n_states)
            ])
        ])
        h = np.vstack([np.ones((2 * reward_dimension, 1)), np.zeros((n_states * (n_actions - 1), 1))])

        solution = solvers.lp(matrix(-c), matrix(G), matrix(h))
        alpha = np.asarray(solution['x'][-reward_dimension:])
        return alpha

    def solvep(self):
        """
        Solve the Inverse RL problem
        :return: Scalarization weights as array.
        """
        n_states, n_actions, reward_dimension, gamma, P, R, pi, P_pi = self._prepare_variables()

        v = self._prepare_v(n_states, n_actions, reward_dimension, P)

        # weights for minimization term
        c = np.vstack([np.ones((n_states, 1)), np.zeros((n_states * (n_actions - 1), 1)), np.zeros((reward_dimension, 1))])
        # big block selection matrix
        G = np.vstack([
            np.hstack([np.zeros((reward_dimension, n_states)), np.zeros((reward_dimension, n_states * (n_actions - 1))), np.eye(reward_dimension)]),
            np.hstack([np.zeros((reward_dimension, n_states)), np.zeros((reward_dimension, n_states * (n_actions - 1))), -np.eye(reward_dimension)]),
            np.hstack([
                np.vstack([self._vertical_ones(n_actions - 1, n_states, i) for i in xrange(n_states)]),
                np.eye(n_states * (n_actions - 1), n_states * (n_actions - 1)),
                np.zeros((n_states * (n_actions - 1), reward_dimension))
            ]),
            np.hstack([
                np.zeros((n_states * (n_actions - 1), n_states)),
                np.eye(n_states * (n_actions - 1), n_states * (n_actions - 1)),
                np.vstack([
                    np.vstack([
                        -v[i, j, :].reshape(1, -1)
                        for j in xrange(n_actions - 1)
                    ])
                    for i in xrange(n_states)
                ])
            ]),
            np.hstack([
                np.zeros((n_states * (n_actions - 1), n_states)),
                2 * np.eye(n_states * (n_actions - 1), n_states * (n_actions - 1)),
                np.vstack([
                    np.vstack([
                        -v[i, j, :].reshape(1, -1)
                        for j in xrange(n_actions - 1)
                    ])
                    for i in xrange(n_states)
                ])
            ])
        ])
        # right hand side of inequalities
        h = np.vstack([np.ones((2 * reward_dimension, 1)), np.zeros(((n_states * (n_actions - 1)) * 3, 1))])

        solution = solvers.lp(matrix(-c), matrix(G), matrix(h))
        alpha = np.asarray(solution['x'][-reward_dimension:])
        return alpha