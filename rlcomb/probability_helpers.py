"""
Created on Feb, 16 2016

@author: Dominik Meyer <meyerd@mytum.de>
"""

"""
This module contains various helper functions with regard
to probability matrices and sampling.
"""

import numpy as np


def assureProbabilityMatrix(P):
    """
    Checks if the matrix P is a valid probability transition matrix.
    This means it has to contain transitions from state i to state j
    in the (i, j)-th entry (i-th column, j-th row).

    :param P: Probability transition matrix
    """
    if P.shape[0] != P.shape[1]:
        raise RuntimeError("Probability matrix check failed: Matrix is not square.")
    psum = P.sum(axis=1)
    if np.abs(psum - 1.0).any() > np.finfo(P.dtype).eps:
        raise RuntimeError("Probability matrix check failed: Columns don't add up to one.")


def assurePolicyMatrix(pi):
    """
    Checks if the matrix P is a valid policy distribution matrix.
    This means it has to contain the probability to choose action a
    under the condition, that the system is in state i in the
    element (a, i) (a-th row, i-th column).

    :param pi: Policy matrix
    """
    psum = pi.sum(axis=0)
    if np.abs(psum - 1.0).any() > np.finfo(pi.dtype).eps:
        raise RuntimeError("Policy matrix check failed: Rows don't add up to one.")
