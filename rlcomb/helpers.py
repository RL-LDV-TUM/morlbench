"""
Created on Sep 14, 2012

@author: Dominik Meyer <meyerd@mytum.de>
"""

"""
This module contains miscellaneous helper
functions that didn't fit in any other place.
"""

import cPickle as pickle
from os.path import isfile
import numpy as np


def virtualFunction():
    raise RuntimeError('Virtual function not implemented.')


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


def loadMatrixIfExists(filename):
    """
    Load a default matrix if the file exists.

    :param filename: Filename were to load from
    :return: Unpickled object
    """
    try:
        with open(filename, 'r') as f:
            ret = pickle.load(f)
    except:
        return None
    return ret


class SaveableObject(object):
    """
    This should be a generic class for custom object
    storage, where you may not want to store all
    data associated with it, but only specific parts
    of the __dict__ of the object.
    """

    def __init__(self, keys, **kwargs):
        """
        Parameters
        ----------
        :param keys: a list of __dict__ entries that
            should be stored, when the object
            is pickled.
        """
        self._saveable_keys_ = keys

#    def __getstate__(self):
#        state = {}
#        for k in self._saveable_keys_:
#            state[k] = self.__dict__[k]
#
#        state['_saveable_keys_'] = self.__dict__['_saveable_keys_']
#        return state
#
#    def __setstate__(self, state):
#        for k, v in state.items():
#            self.__dict__[k] = v

    def save(self, filename, overwrite=False):
        """
        Save the object to disk using cPickle.

        :param filename: Filename where to save the serialized object.
        :param overwrite: Overwrite existing files.
        """
        if isfile(filename) and not overwrite:
            raise RuntimeError('file %s already exists.' % (filename))
        f = open(filename, 'wb')
        if not f:
            raise RuntimeError('could not open %s' % (filename))
        state = {}
        for k in self._saveable_keys_:
            state[k] = self.__dict__[k]
        state['_saveable_keys_'] = self.__dict__['_saveable_keys_']
        pickle.dump(state, f)
        f.close()

    def load(self, filename):
        """
        Load pickled object from file.

        :param filename: Filename where to load the object from.
        """
        if not isfile(filename):
            raise RuntimeError('file %s does not exist' % (filename))
        f = open(filename, 'rb')
        if not f:
            raise RuntimeError('could not open %s' % (filename))
        state = pickle.load(f)
        f.close()
        self.__dict__.update(state)\
