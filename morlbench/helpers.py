#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sep 14, 2012

This module contains miscellaneous helper
functions that didn't fit in any other place.

@author: Dominik Meyer <meyerd@mytum.de>
"""

import cPickle as pickle
from os.path import isfile
import numpy as np
import math
import time
from functools import wraps
import sys
from matplotlib.mlab import PCA as mlabPCA

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


def remove_duplicates(array):

    array = [tuple(element) for element in array]
    removed = set(array)
    removed_array = [elem for elem in removed]
    return removed_array


class HyperVolumeCalculator:
    """
    this class computes hypervolumes
    @author Simon Wölzmüller <ga35voz@mytum.de>

    """
    def __init__(self, ref_point):
        """
        initializes calculator
        :param ref_point: reference point which the volume is referred to
        :return: value of hypervolume
        """
        self.ref_point = ref_point
        self.list = []
        self.list_compressing = False

    def extract_front(self, given_points):
        """
        searches pareto front of given point set
        :param given_points: all points, array of arrays(d-dimensional)
        :return: array of arrays(d-dimensional) pareto points
        """
        dimensions = len(self.ref_point)
        # no sense for 1 dimension
        if dimensions == 1:
            return 0
        # special algorithm for 2 dimensions
        if dimensions == 2:
            front = self.pareto_optimal_front_2_dim(given_points)
            return front
        # for all other dimensions
        if dimensions > 2:
            front = self.pareto_optimal_front_d_dim(given_points)
            return front

    def pareto_optimal_front_2_dim(self, points):
        """
        this function extracts pareto front from 2 dim data set
        :param points: 2d - data - set
        :return: pareto front
        """
        # sort first dimension
        points = np.vstack((points))
        points = sorted(points, key=lambda y: y[0])[::-1]
        # add the first point
        pareto_front = []
        pareto_front.append(points[0])
        # test other points in missing dimension
        for pair in points[1:]:
            # append point to pareto front if it dominates other points
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        return pareto_front

    def pareto_optimal_front_d_dim(self, points):
        """
        this function extracts pareto optimal front from d dimensional data set
        :param points: d dimensional data points which should be analysed
        :return: pareto front of the input point set
        """
        # sort first dimension:
        if len(points):
            points = np.vstack((points))
            points = sorted(points, key=lambda y: y[0])[::-1]
            # add first point to pareto front
            pareto_front = points[0:1][:]
            # test next point
            for point in points[1:][:]:
                # if all components of candidate are greater than the corresponding  components of the first element
                if sum([point[x] >= pareto_front[-1][x] for x in xrange(len(point))]) == len(point):
                    # add that element to the front
                    pareto_front = np.concatenate((pareto_front, [point]))
            return pareto_front
        else:
            return 0

    def dominates(self, p, q, k=None):
        # finds out if some point dominates another
        if k is None:
            k = len(p)
        d = True
        # test every dimension
        while d and k < len(p):
            d = not (q[k] > p[k])
            k += 1
        return d

    def compute_hv(self, point_set):
        """
        computes hypervolume. at first it searches pareto front and selects points that are dominating refpoint
        this function is a slightly adapted version of While et al HSO - ALgorithm, a hypervolume calculator:

        """


        def insert(p, k, pl):
            ql = []
            while pl and pl[0][k] > p[k]:
                ql.append(pl[0])
                pl = pl[1:]
            ql.append(p)
            while pl:
                if not self.dominates(p, pl[0], k):
                    ql.append(pl[0])
                pl = pl[1:]
            return ql

        def slice(pl, k, ref):
            p = pl[0]
            pl = pl[1:]
            ql = []
            s = []
            while pl:
                ql = insert(p, k + 1, ql)
                p_prime = pl[0]
                s.append((math.fabs(p[k] - p_prime[k]), ql))
                p = p_prime
                pl = pl[1:]
            ql = insert(p, k + 1, ql)
            s.append((math.fabs(p[k] - ref[k]), ql))
            return s

        # reference point
        reference_point = self.ref_point
        # rel points
        relevant_points = point_set
        if len(relevant_points) == 0:
            return 0.0

        if reference_point:
            for point in relevant_points:
                # only consider points that dominate the reference point
                if self.dominates(point, reference_point, None):
                    np.append(relevant_points, point)

        # compute the hypervolume
        if self.list_compressing:
            ps = remove_duplicates(relevant_points)
        else:
            ps = relevant_points
        # store reference point
        ref = self.ref_point
        # find out dimension
        n = min([len(p) for p in ps])
        # if no reference is given, compute it with given points (max in each dimension)
        if ref is None:
            ref = [max(ps, key=lambda x: x[o])[o] for o in range(n)]
        # sort the points in first dimension and reverse
        pl = np.vstack(ps[:])
        pl = sorted(pl, key=lambda q: q[0])[::-1]
        s = [(1, pl)]
        for k in xrange(n - 1):
            s_prime = []
            for x, ql in s:
                # slice the volume in little volumes
                for x_prime, ql_prime in slice(ql, k, ref):
                    s_prime.append((x * x_prime, ql_prime))
            s = s_prime
        vol = 0
        # add the volumes
        for x, ql in s:
            vol = vol + x * math.fabs(ql[0][n - 1] - ref[n - 1])
        return vol


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer


def compute_hull(points):
    """
    Hull computation over PCA (principle components analysis)
    :param points:
    :return:
    """
    raise NotImplementedError("Hull computation over PCA not yet ready")
    final_points = []
    if len(points) <= 1:
        return 0, 0, points

    mlab_PCA = mlabPCA(points)
    # cutoff = 1e-10
    # # compute the vectors mean
    # points = [np.array(points[i]) for i in xrange(len(points))]
    # mean = [np.mean([points[u][j] for u in xrange(len(points))]) for j in xrange(len(points[0]))]
    # # subtract the mean of each vector
    # submean = [p - mean for p in points]
    #
    # # build covariance matrix
    # covar = np.zeros((len(points[0]), len(points[0])))
    # for v1 in xrange(len(submean)):
    #     for dim1 in xrange(len(submean[v1])):
    #         for dim2 in xrange(len(submean[v1])):
    #             covar[dim1][dim2] += submean[v1][dim2]*submean[v1][dim1]
    # # ... and its eigenvalues and eigenvectors
    # values, vec = np.linalg.eig(covar)
    # # count of values that are significant
    # count = len(values[abs(values) > cutoff])
    # if count == 0:
    #     # no significant eigenvalue
    #     final_points = [[0, ]*len(points)]
    # axes = np.zeros((count, len(points[0])))
    # i = 0
    # for val in xrange(len(values)):
    #     if values[val] <= cutoff:
    #         continue
    #     for j in xrange(len(points[0])):
    #         axes[i, j] = -vec[i, j]
    #     i += 1
    # for ax in xrange(len(axes)):
    #     for dim in xrange(len(axes[ax])):
    #         final_points = axes

    # for v1 in xrange(len(submean)):
    #     new_points = 0
    # new_points = [nep[0] for nep in new_points]
    # if type(new_points[0]) == float or type(new_points[0]) == int:
    #      new_points = [[new_points[i]] for i in xrange(len(new_points))]
    #
    # return mean, count, new_points
    return 0



