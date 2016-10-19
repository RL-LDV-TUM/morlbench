#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sep 14, 2012

This module contains miscellaneous helper
functions that didn't fit in any other place.

@author: Dominik Meyer <meyerd@mytum.de>
@author: Simon Wölzmüller <ga35voz@mytum.de>

    Copyright (C) 2016  Dominik Meyer, Simon Wölzmüller

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

import cPickle as pickle
from os.path import isfile
import numpy as np
import math
import time
from functools import wraps
import sys
from matplotlib.mlab import PCA as mlabPCA
from scipy.spatial import ConvexHull

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
    if type(array[0]) is int or type(array[0]) is float:
        array = tuple(array)
        removed = set(array)
        removed_array = [elem for elem in removed]
        return removed_array
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
    @author Simon Woelzmueller <ga35voz@mytum.de>
    :param points: the points we need the hull to
    :return: the convex hull
    """
    # find out dimension
    points_dimension = len(points[0])
    def all_same(items):
        return all(x == items[0] for x in items)

    points = remove_duplicates(points)
    if len(points) <= len(points[0])+2:
        return points
    # dim with same points reduction
    dim = []
    for i in xrange(points_dimension):
        column = np.array([points[u][i] for u in xrange(len(points))])
        if all_same(column):
            dim.append(i)
    if points_dimension - len(dim) <= 1:
        conv_hull = []
        for i in xrange(points_dimension):
            if not dim.count(i):
                column = [points[u][i] for u in xrange(len(points))]
                maxind = column.index(max(column))
                minind = column.index(min(column))
                conv_hull.append(points[maxind])
                conv_hull.append(points[minind])
        return conv_hull
    if points_dimension - len(dim) <= 2:
        temp = []
        for u in xrange(len(points)):
            temp_point = []
            for i in xrange(points_dimension):
                if not dim.count(i):
                    temp_point.append(points[u][i])
            temp.append(temp_point)

        temp = np.array([np.array(p) for p in temp])
        if [all_same(temp[i]) for i in xrange(len(temp))].count(True) == len(temp):
            conv_hull = []
            column = [points[u][0] for u in xrange(len(points))]
            maxind = column.index(max(column))
            minind = column.index(min(column))
            conv_hull.append(points[maxind])
            conv_hull.append(points[minind])

            return conv_hull
        hull = ConvexHull(temp)
        hullp = [points[i] for i in hull.vertices]
        hull.close()
        return np.array(hullp)

    if len(points) <= len(points[0]) + 2:
        return points
    points = np.array([np.array(p) for p in points])
    hull = ConvexHull(points)
    hullp = [points[i] for i in hull.vertices]
    hull.close()

    return np.array(hullp)



