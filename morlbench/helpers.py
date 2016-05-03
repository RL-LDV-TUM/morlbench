#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
        self.__dict__.update(state)


class HyperVolumeCalculator:
    """
    this class computes hypervolumes, it is an adaption on the paper: "An improved dimension sweep algorithm
    for the hypervolume indicator (Fonseca,Paquete,Ibanez)
    """
    def __init__(self, ref_point, point_set):
        """
        initializes calculator
        :param ref_point: reference point which the volume is referred to
        :return: value of hypervolume
        """
        self.ref_point = ref_point
        self.point_set = self.extract_front(point_set)

    def extract_front(self, given_points):
        """
        searches pareto front of given point set
        :param given_points: all points, array of arrays(d-dimensional)
        :return: array of arrays(d-dimensional) pareto points
        """
        dimensions = given_points.shape[1]
        # no sense for 1 dimension
        if dimensions == 1:
            return 0
        # special algorithm for 2 dimensions
        if dimensions == 2:
            front = self.pareto_front_2_dim(given_points)
        # for all other dimensions
        if dimensions > 2:
            front = self.pareto_front_d_dim(given_points)
        return front

    def pareto_front_2_dim(self, points):
        """
        this function extracts pareto front from 2 dim data set
        recipe adapted on Jamie Bull (MIT)
        :param points: 2d - data - set
        :return: pareto front
        """
        # sort first dimension
        points = points[points[:, 0].argsort()][::-1]
        # add the first dimension(yet sorted)
        pareto_front = []
        pareto_front.append([points[0][0], points[0][1]])
        # test other dimension in pairs
        for pair in points[1:]:
            # append point to pareto front if it dominates other points
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append([pair[0], pair[1]])
        return pareto_front[:]

    def pareto_front_d_dim(self, points):
        """
        this function extracts pareto front from d dimensional data set
        recipe adapted on Jamie Bull (MIT)
        :param points: d dimensional data points which should be analysed
        :return: pareto front of the input point set
        """
        # sort first dimension:
        points = points[points[:, 0].argsort()][::-1]
        # add first dimension to pareto front
        pareto_front = points[0:1, :]
        # test next dimension
        for dimension in points[1:, :]:
            if sum([dimension[x] > pareto_front[-1][x] for x in xrange(len(dimension))]) >= len(dimension):
                pareto_front = np.concatenate((pareto_front, [dimension]))
        return pareto_front

    #TODO: calculate HV!!!

