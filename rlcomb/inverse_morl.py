#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import SaveableObject, virtualFunction
from dynamic_programming import MORLDynamicProgrammingInverse, MORLDynamicProgrammingPolicyEvaluation

import numpy as np
import numpy.linalg as npla
import logging as log
try:
    from cvxopt import matrix, solvers
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
        # transition matrix P_{ss'}
        P = self._problem.P
        # reward matrix R(s)
        R = self._problem.R
        # Policy pi
        pi = self._policy.get_pi()

        # calculate state transition probability matrix together with policy
        P_pi = P * pi[:, :, np.newaxis]
        P_pi = P_pi.sum(axis=1)
        P_pi /= P_pi.sum(axis=1)[:, np.newaxis]
        return (n_states, n_actions, reward_dimension, gamma, P, R, pi, P_pi)

    def _prepare_v(self, n_states, n_actions, reward_dimension, P):
        problem, policy = self._problem, self._policy

        #dp = MORLDynamicProgrammingInverse(problem, policy)
        dp = MORLDynamicProgrammingPolicyEvaluation(problem, policy)
        # V = dp.solve(vector_implementation=True)
        V = dp.solve(vector_implementation=False)

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
        return alpha.ravel()

    def solve_sum_1(self):
        """
        Solve the Inverse RL problem
        :return: Scalarization weights as array.
        """
        n_states, n_actions, reward_dimension, gamma, P, R, pi, P_pi = self._prepare_variables()

        v = self._prepare_v(n_states, n_actions, reward_dimension, P)

        c = -np.vstack([np.ones((n_states, 1)), np.zeros((reward_dimension, 1))])
        G = np.vstack([
            np.hstack([np.zeros((reward_dimension, n_states)), -np.eye(reward_dimension)]),
            np.vstack([
                np.vstack([
                   np.hstack([self._vertical_ones(1, n_states, i), -v[i, j, :].reshape(1, -1)]) for j in xrange(n_actions - 1)
                ]) for i in xrange(n_states)
            ])
        ])
        A = np.hstack([
            np.zeros((1, n_states)), np.ones((1, reward_dimension))
        ])
        b = np.ones((1, 1))
        h = np.vstack([np.zeros((reward_dimension, 1)), np.zeros((n_states * (n_actions - 1), 1))])

        solution = solvers.lp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b))
        alpha = np.asarray(solution['x'][-reward_dimension:])
        return alpha.ravel()

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
        #h = np.vstack([np.ones((2 * reward_dimension, 1)), np.zeros(((n_states * (n_actions - 1)) * 3, 1))])

        h = np.vstack([np.ones((reward_dimension, 1)), np.zeros((reward_dimension, 1)), np.zeros(((n_states * (n_actions - 1)) * 3, 1))])

        solution = solvers.lp(matrix(-c), matrix(G), matrix(h))
        alpha = np.asarray(solution['x'][-reward_dimension:])
        return alpha.ravel()

    def solvealge(self):
        """
        Solve the Inverse RL problem
        :return: Scalarization weights as array.
        """
        n_states, n_actions, reward_dimension, gamma, P, R, pi, P_pi = self._prepare_variables()

        v = self._prepare_v(n_states, n_actions, reward_dimension, P)

        D = reward_dimension
        x_size = n_states + (n_actions - 1) * n_states * 2 + D

        c = -np.hstack([np.ones(n_states), np.zeros(x_size - n_states)])
        assert c.shape[0] == x_size

        A = np.hstack([
                np.zeros((n_states * (n_actions - 1), n_states)),
                np.eye(n_states * (n_actions - 1)),
                -np.eye(n_states * (n_actions - 1)),
                np.vstack([v[i, j, :].reshape(1, -1) for i in range(n_states)
                                                     for j in range(n_actions - 1)])])
        assert A.shape[1] == x_size

        b = np.zeros(A.shape[0])

        bottom_row = np.vstack([
                        np.hstack([
                            np.ones((n_actions - 1, 1)).dot(np.eye(1, n_states, l)),
                            np.hstack([-np.eye(n_actions - 1) if i == l
                                       else np.zeros((n_actions - 1, n_actions - 1))
                                       for i in range(n_states)]),
                            np.hstack([2 * np.eye(n_actions - 1) if i == l
                                       else np.zeros((n_actions - 1, n_actions - 1))
                                       for i in range(n_states)]),
                            np.zeros((n_actions - 1, D))])
                        for l in range(n_states)
                        ])
        assert bottom_row.shape[1] == x_size

        G = np.vstack([
                np.hstack([
                    np.zeros((D, n_states)),
                    np.zeros((D, n_states * (n_actions - 1))),
                    np.zeros((D, n_states * (n_actions - 1))),
                    np.eye(D)]),
                np.hstack([
                    np.zeros((D, n_states)),
                    np.zeros((D, n_states * (n_actions - 1))),
                    np.zeros((D, n_states * (n_actions - 1))),
                    -np.eye(D)]),
                np.hstack([
                    np.zeros((n_states * (n_actions - 1), n_states)),
                    -np.eye(n_states * (n_actions - 1)),
                    np.zeros((n_states * (n_actions - 1), n_states * (n_actions - 1))),
                    np.zeros((n_states * (n_actions - 1), D))]),
                np.hstack([
                    np.zeros((n_states * (n_actions - 1), n_states)),
                    np.zeros((n_states * (n_actions - 1), n_states * (n_actions - 1))),
                    -np.eye(n_states * (n_actions - 1)),
                    np.zeros((n_states * (n_actions - 1), D))]),
                bottom_row
        ])
        assert G.shape[1] == x_size

        # h = np.vstack([np.ones((D * 2, 1)),
        #                np.zeros((n_states * (n_actions - 1) * 2 + bottom_row.shape[0], 1))])
        h = np.vstack([10.0 * np.ones((D, 1)), np.zeros((D, 1)),
                       np.zeros((n_states * (n_actions - 1) * 2 + bottom_row.shape[0], 1))])

        # c = c.reshape(-1, 1)
        # b = b.reshape(-1, 1)
        # print c.shape
        # print G.shape
        # print h.shape
        # print A.shape
        # print b.shape

        # normalize each row
        # asum = np.sum(np.abs(A), axis=1)
        # print asum.shape
        # print A.shape
        # A /= asum[:, np.newaxis]
        # b /= asum

        # solvers.options['feastol'] = 1e-1
        # solvers.options['abstol'] = 1e-3
        # solvers.options['show_progress'] = True
        # c = matrix(c)
        # G = matrix(G)
        # h = matrix(h)
        # A = matrix(A)
        # b = matrix(b)
        solution = solvers.lp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b))
        alpha = np.asarray(solution['x'][-reward_dimension:], dtype=np.double)
        return alpha.ravel()
