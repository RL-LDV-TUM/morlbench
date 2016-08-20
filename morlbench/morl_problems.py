#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
@author: Johannes Feldmaier <johannes.feldmaier@tum.de>

"""

from helpers import SaveableObject, loadMatrixIfExists, virtualFunction
from probability_helpers import assureProbabilityMatrix, sampleFromDiscreteDistribution

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sqrt
import logging as log
import random
import sys

import os

my_debug = log.getLogger().getEffectiveLevel() == log.DEBUG


class MORLProblem(SaveableObject):
    def __init__(self, *args, **kwargs):
        super(MORLProblem, self).__init__(args, **kwargs)

    def reset(self):
        virtualFunction()

    def _construct_r(self):
        # Multi objective reward has to be stationary for the batch IRL algorithms
        # That means a reward that grows with the number of steps is difficult to
        # handle.

        self.R = np.zeros((self.n_states, self.reward_dimension))
        for i in xrange(self.n_states):
            self.R[i, :] = self._get_reward(i)

    def _get_reward(self, state):
        virtualFunction()

    def __str__(self):
        return self.__class__.__name__

    def play(self, action):
        virtualFunction()


class Deepsea(MORLProblem):
    """
    This class represents a Deepsea problem.
    All the parameters should be set up on object
    creation. Then the Deepsea problem can be used
    iteratively by calling "action".
    """

    def __init__(self, scene=None, actions=None, gamma=0.9, state=0, extended_reward=False):
        """
        Initialize the Deepsea problem.

        Parameters
        ----------
        :param scene: array, Map of the deepsea landscape. Entries represent
            rewards. Invalid states get a value of "-100" (e.g. walls, ground).
            Positive values correspond to treasures.
        :param actions: The name of the actions: Here the directions the
            submarine can move - left, right, up, down.
        :param gamma: The discount factor of the problem.
        """

        super(Deepsea, self).__init__(
            ['state', '_time', '_actions', '_scene'])

        self._time = 0

        self._start_state = state
        self.P = None
        self.R = None
        # Discount Factor
        self.gamma = gamma

        if actions is None:
            # Default actions
            # actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
            actions = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))
            # if an idle action is required
            # actions = (np.array([-1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, -1]), np.array([0, 0]))

        if scene is None:
            # Empty _scene array - no ground
            self._scene = np.zeros((11, 10))

            # Default Map as used in general MORL papers
            self._scene[2:11, 0] = -100
            self._scene[3:11, 1] = -100
            self._scene[4:11, 2] = -100
            self._scene[5:11, 3:6] = -100
            self._scene[8:11, 6:8] = -100
            self._scene[10, 8] = -100
            # Rewards of the default map
            self._scene[1, 0] = 1
            self._scene[2, 1] = 2
            self._scene[3, 2] = 3
            self._scene[4, 3] = 5
            self._scene[4, 4] = 8
            self._scene[4, 5] = 16
            self._scene[7, 6] = 24
            self._scene[7, 7] = 50
            self._scene[9, 8] = 74
            self._scene[10, 9] = 124

            # Diagonal Map
            # self._scene[2:11, 0] = -100
            # self._scene[3:11, 1] = -100
            # self._scene[4:11, 2] = -100
            # self._scene[5:11, 3] = -100
            # self._scene[6:11, 4] = -100
            # self._scene[7:11, 5] = -100
            # self._scene[8:11, 6] = -100
            # self._scene[9:11, 7] = -100
            # self._scene[10, 8] = -100

            # Normalized reward
            # self._scene[1, 0] = 1/124.0
            # self._scene[2, 1] = 2/124.0
            # self._scene[3, 2] = 3/124.0
            # self._scene[4, 3] = 5/124.0
            # self._scene[4, 4] = 8/124.0
            # self._scene[4, 5] = 16/124.0
            # self._scene[7, 6] = 24/124.0
            # self._scene[7, 7] = 50/124.0
            # self._scene[9, 8] = 74/124.0
            # self._scene[10, 9] = 124/124.0

        # old flat map including ground states
        # self._flat_map = np.ravel(self._scene, order='C')  # flat map with C-style order (column-first)
        # self.n_states = (self._scene.shape[0] * self._scene.shape[1]) + 1  # +1 for terminal state

        self._flat_map = np.argwhere(self._scene >= 0)  # get all indices greater than zero
        # get all elements greater than zero and stack them to the corresponding index
        self._flat_map = np.column_stack((self._flat_map,  self._scene[self._scene >= 0]))
        self.n_states = len(self._flat_map) + 1  # +1 for terminal state

        self.n_states_print = self.n_states - 1
        self._index_terminal_state = self.n_states - 1

        self.actions = actions
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions

        self.reward_dimension = 2
        self._extended_reward = extended_reward
        if extended_reward:
            self.reward_dimension += self.n_states
            # self.reward_dimension = self.n_states

        self.reset()

        # build state transition matrix P_{ss'} where (i, j) is the transition probability
        # from state i to j
        if self.P is None:
            self._construct_p()

        # build reward vector R(s)
        if self.R is None:
            self._construct_r()

    def name(self):
        return "Deepsea"

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self._scene.shape[0]):
            for j in xrange(self._scene.shape[1]):
                pos = (i, j)
                pos_index = self._get_index(pos)
                if pos_index >= 0:  # update p only if it is a valid state
                    for a in xrange(self.n_actions):  # for all action except the last -> idle action
                        n_pos = pos + self.actions[a]
                        n_pos_index = self._get_index(n_pos)

                        if self._in_map(n_pos) and self._flat_map[pos_index, 2] == 0 and n_pos_index >= 0:
                            # we are in the map and no special state
                            if self._flat_map[n_pos_index, 2] >= 0:  # normal or reward _next_ state
                                self.P[pos_index, a, n_pos_index] = 1.0
                            elif self._flat_map[n_pos_index, 2] < 0:  # we go directly into the ground
                                self.P[pos_index, a, pos_index] = 1.0  # stay at position
                            else:
                                raise ValueError('Sollte nicht vorkommen (state: %i)!', pos_index)
                        # state must be a ground or reward state -> special transition
                        elif self._flat_map[pos_index, 2] < 0:  # current state is ground -> we stay there
                            self.P[pos_index, a, pos_index] = 1.0
                        elif self._flat_map[pos_index, 2] > 0:  # reward state -> we transfer to the terminal state
                            self.P[pos_index, a, self._index_terminal_state] = 1.0
                        else:
                            # we are out of the map and stay in our state
                            self.P[pos_index, a, pos_index] = 1.0

                        # idle action -> we always stay in our state except for reward states -> terminal state
                        # if self._flat_map[pos_index, 2] > 0:
                        #     self.P[pos_index, -1, self._index_terminal_state] = 1.0
                        # else:
                        #     self.P[pos_index, -1, pos_index] = 1.0
        # stay in terminal state forever
        self.P[-1, :, -1] = 1.0

    def reset(self):
        self.state = self._start_state
        self.terminal_state = False
        self.treasure_state = False
        self._time = 0
        self.last_state = self.state
        self._position = self._get_position(self.state)
        self._last_position = self._position

    @property
    def scene_x_dim(self):
        return self._scene.shape[1]

    @property
    def scene_y_dim(self):
        return self._scene.shape[0]

    def _get_index(self, position):
        if self._in_map(position):
            # return np.ravel_multi_index(position, self._scene.shape)
            index = np.argwhere((self._flat_map[:, [0, 1]] == position).all(-1))
            if index.size:
                return np.asscalar(index)
            else:
                if my_debug:
                    log.debug('Invalid position ' + str(position) + '-> out of valid map')
                return -1
        else:
            if my_debug:
                log.debug('Error: Position out of map!')
            return -1

    def _get_position(self, index):
        if index < self.n_states - 1:
            return self._flat_map[index, [0, 1]]
        else:
            if my_debug:
                log.debug('Error: Index out of list!')
            return -1

        # if index < (self._scene.shape[0] * self._scene.shape[1]):
        #     return np.unravel_index(index, self._scene.shape)
        # else:
        #     if my_debug: log.debug('Error: Index out of list!')
        #     return -1

    def _in_map(self, position):
        return not ((position[0] < 0) or (position[0] > self._scene.shape[0] - 1) or (position[1] < 0) or
                    (position[1] > self._scene.shape[1] - 1))

    def print_map(self, pos=None):
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()

    def _get_reward(self, state):
        r = np.zeros(self.reward_dimension)

        if state == self._index_terminal_state:
            r[0] = 0.0
            r[1] = 0.0
        else:
            r[1] = -1.0
            map_value = self._flat_map[state, 2]
            if map_value > 0:
                r[0] = map_value
            elif map_value < 0:
                r[0] = 0.0
            elif map_value == 0:
                r[0] = 0.0
            else:
                raise ValueError('Invalid map_value for state %i', state)

        if self._extended_reward:
            r[state + 2] = 1.0

        return r

    def play(self, action):
        """
        Perform an action with the submarine
        and receive reward (or not).

        Parameters
        ----------
        action: integer, Which action will be chosen
            the agent. (0: left, 1: right, 2: up, 3: down).

        Returns
        -------
        reward: reward of the current state.
        """

        self._time += 1

        last_position = np.copy(self._position)  # numpy arrays are mutable -> must be copied

        if self.terminal_state:
            self.last_state = self.state
            return self._get_reward(self.state)

        if self.treasure_state:
            self.last_state = self.state
            self.state = self._index_terminal_state
            self.terminal_state = True
            return self._get_reward(self.state)

        # check if in map and if the following state is a ground (valid) state (index = -1)
        if self._in_map(self._position + self.actions[action]) and self._get_index(self._position +
                                                                                   self.actions[action]) >= 0:
            self._position += self.actions[action]
            map_value = self._flat_map[self._get_index(self._position), 2]
            if my_debug:
                log.debug('Moved from pos ' + str(last_position) + ' by ' + str(self.actions[action]) +
                          ' to pos: ' + str(self._position) + ')')
            if map_value < 0:
                self._position = last_position
                if my_debug:
                    log.debug('Ground touched!')
            elif map_value > 0:
                if my_debug:
                    log.debug('Treasure found! - I got a reward of ' + str(map_value))
                self.treasure_state = True
            else:
                if my_debug:
                    log.debug('Normal state!')
        else:
            if my_debug:
                log.debug('Move not allowed! -> out of map')

        if my_debug:
            log.debug('New position: ' + str(self._position))

        self._last_position = np.copy(last_position)
        self.last_state = self.state
        self.state = self._get_index(self._position)

        return self._get_reward(self.state)


class DeepseaEnergy(Deepsea):
    def __init__(self, scene=None, actions=None, state=0, energy=200):
        """
        energy: integer > 0, Amount of energy the
            the submarines battery is loaded.
        """
        self._energy = energy
        self._init_energy = energy
        self.reward_dimension = 3

        super(DeepseaEnergy, self).__init__(scene=scene, actions=actions, state=state)
        super(Deepsea, self).__init__(keys=['_time', 'actions', '_scene', '_energy'])

    def reset(self):
        super(DeepseaEnergy, self).reset()
        self._energy = self._init_energy

    def play(self, action):
        reward = super(DeepseaEnergy, self).play(action)
        self._energy -= 1
        # return np.array([reward, -self._time, self._energy])
        return np.array([reward, -1, self._energy])


class MountainCar(MORLProblem):
    def __init__(self, acc_fac=0.00001, cf=0.0002, time_lim=30, state=38, gamma=0.9):
        """
        Initialize the Mountain car problem.

        Parameters
        ----------
        state: default state is -0.5
        """

        super(MountainCar, self).__init__(
                ['state', '_time', 'actions', '_scene'])

        self.actions = ('left', 'right', 'none')
        self.n_actions = 3

        self.n_actions_print = self.n_actions - 1
        self.P = None
        # Discount Factor
        self.gamma = gamma
        self.init_state = state
        self._nb_actions = 0  # counter for acceration actions
        self._minPosition = -1.2  # Minimum car position
        self._maxPosition = 0.6  # Maximum car position (past goal)
        self._maxVelocity = 0.07  # Maximum velocity of car
        self._goalPosition = 0.6  # Goal position - how to tell we are done
        self._accelerationFactor = acc_fac  # discount for accelerations
        self._maxGoalVelocity = 0.07
        self.n_states = 200  # for state discretization
        self.state_solution = (self._maxPosition - self._minPosition)/self.n_states  # continouus step for one discrete
        self._start_state = state # initial position ~= -0.5 ^= 38
        self._velocity = 0  # start velocity
        self.state = state      # initialize state variable
        self.last_state = self.state       # at the beginning we have no other state
        self._position = self.get_position(self._start_state)   # the corresponding continous position on landscape
        self._time = 0      # time variable
        self._default_reward = 100      # reward for reaching goal position
        self.terminal_state = False     # variable for reaching terminal state
        self.n_states_print = self.n_states - 1
        self._goalState = self.n_states-1
        self.reward_dimension = 3
        self.time_token = []
        self.cosine_factor = cf
        # self._construct_p()
        self.acceleration = 0

    def reset(self):
        self._velocity = 0
        self.state = self._start_state
        self.last_state = self._start_state
        self._position = self.get_position(self._start_state)
        self._time = 0
        self._nb_actions = 0
        self.terminal_state = False

    def distance(self, point1, point2):
        return np.abs(point1-point2)

    def get_position(self, state):
        return self._minPosition+state*self.state_solution

    def get_state(self, position):
        difference = position - self._minPosition
        rounded_state = int(difference/self.state_solution)
        state_candidates = [rounded_state-1, rounded_state, rounded_state+1]
        distances = dict()
        for i in state_candidates:
            distances[i] = self.distance(self.get_position(i), position)
        for u in distances.keys():
            if distances[u] == min(distances.values()):
                if u > self.n_states-1:
                    # to avoid size overflow in qtable
                    u = self.n_states-1
                return u

    def name(self):
        return "Mountain Car"

    def play(self, action):
        """
        Perform an action with the car in the mountains
        and receive reward (or not).

        Parameters
        ----------
        action: integer, Which action will be chosen
            0: no action -> coasting
            1: forward thrust
            -1: backward thrust

        """

        # Remember state before executing action
        self.last_state = self.state
        # t
        self._time += 1
        # we use a mapping for indices
        map_actions = {
            'none': 0,  # coasting
            'right': 1,  # forward thrust
            'left': -1,  # backward thrust
        }
        # Determine acceleration factor
        if action < len(self.actions):
            # factor is a variable +1 for accelerating, -1 one for reversing, 0 for nothing
            factor = map_actions[self.actions[action]] # map action to thrust factor
            self.acceleration = factor
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0  # Default action
        # apply that factor on the car movement
        self.car_sim(factor)
        return self._get_reward(self.state)

    def _get_reward(self, state):
        reward = np.zeros(self.reward_dimension)
        # check if we reached goal
        if self.terminal_state:
            # reward!
            reward[0] = self._default_reward
        if self.acceleration == 1:
            reward[1] = -1
        if self.acceleration == -1:
            reward[2] = -2
        return reward

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            xi, yi = self.get_position(i)
            for a in xrange(self.n_actions):
                ox, oy = self.actions[a]
                tx, ty = xi + ox, yi + oy

                if not self._in_map((tx, ty)):
                    self.P[i, a, i] = 1.0
                else:
                    j = self._get_index((tx, ty))
                    self.P[i, a, j] = 1.0

    def car_sim(self, factor):

        def minmax(val, lim1, lim2):
            """
            Bounding item between lim1 and lim2

            :param val:
            :param lim1:
            :param lim2:
            :return:
            """
            return max(lim1, min(lim2, val))

        # State update
        velocity_change = self._accelerationFactor * factor - self.cosine_factor * cos(3 * self._position)
        # compute velocity
        self._velocity = minmax(self._velocity + velocity_change, -self._maxVelocity, self._maxVelocity)
        # add that little progress to position
        self._position += self._velocity
        # look if we've gone too far
        self._position = minmax(self._position, self._minPosition, self._maxPosition)
        # check in the next discrete state
        self.state = self.get_state(self._position)
        # check if we're on the wrong side
        if self._position <= self._minPosition:  # and (self._velocity < 0)
            # inelastic wall stops the car imediately
            self._velocity = 0.0
        # check if we've reached the goal
        if self.state >= self._goalState:
            self.terminal_state = True
            # store time token for this
            self.time_token.append(self._time)


class MountainCarTime(MountainCar):
    """
    reward: [time, reversal, front]
    every time step: time=-1
    every reversal: reversal=-1
    every front acceleration: front = -1
    reaching goal position: time = 100
    """
    def __init__(self, acc_fac=0.00001, cf = 0.0002, time_lim=30, state=50, gamma=0.9):
        """
        Initialize the Multi Objective Mountain car problem.

        Parameters
        ----------
        state: default state is -0.5
        """
        self._nb_actions = 0
        super(MountainCarTime, self).__init__(state=state, acc_fac=acc_fac, cf=cf, time_lim=time_lim)
        self.reward_dimension = 3

    def reset(self):
        self._velocity = 0
        self.state = self._start_state
        self.last_state = self._start_state
        self._position = self.get_position(self._start_state)
        self._time = 0
        self._nb_actions = 0
        self.terminal_state = False

    def play(self, action):
        """
        Perform an action with the car in the
        multi objective mountains and receive reward (or not).

        Multi objectives: Minimize Time and accelerating Actions.

        Parameters
        ----------
        action: integer, Which action will be chosen
            0: no action -> coasting
            1: forward thrust
            -1: backward thrust

        Returns
        -------
        reward: reward of the current state.
        """

        # Remember state before executing action
        self.last_state = self.state

        self._time += 1

        map_actions = {
            'left': -1,  # backward thrust
            'right': 1,  # forward thrust
            'none': 0,  # coasting
        }
        # Determine acceleration factor
        if action < len(self.actions):
            factor = map_actions[self.actions[action]]  # map action to thrust factor
            if (self.actions[action] == 'right') or (self.actions[action] == 'left'):
                self._nb_actions += 1
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0  # Default action

        self.car_sim(factor)
        return self._get_reward(self.state)

    def _get_reward(self, state):
        reward = np.zeros(self.reward_dimension)
        if self._nb_actions >= 20:
            reward[2] = -1

        if self.terminal_state:
            reward[0] = self._default_reward
        else:
            reward[1] = -1
        return reward


class Gridworld(MORLProblem):
    """
    Original Simple-MORL-Gridworld.
    """

    def __init__(self, size=10, gamma=0.9):
        self.gamma = gamma

        self.actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        self.n_states = size * size
        self.n_states_print = self.n_states
        self._size = size
        self.reward_dimension = self.n_states
        self.init_state = 0
        self.P = None
        self.R = None

        # Default Map as used in general MORL papers
        self._scene = np.zeros((size, size))
        self._scene[0, size-1] = 1
        self._scene[size-1, 0] = 1
        self._scene[size-1, size-1] = 1

        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            xi, yi = self._get_position(i)
            for a in xrange(self.n_actions):
                ox, oy = self.actions[a]
                tx, ty = xi + ox, yi + oy

                if not self._in_map((tx, ty)):
                    self.P[i, a, i] = 1.0
                else:
                    j = self._get_index((tx, ty))
                    self.P[i, a, j] = 1.0

    def reset(self):
        self.state = 0
        self.last_state = 0
        self.terminal_state = False

    def _get_index(self, position):
        # return position[1] * self.scene_x_dim + position[0]
        return position[0] * self.scene_x_dim + position[1]

    def _get_position(self, index):
        return index // self.scene_x_dim, index % self.scene_x_dim

    def _in_map(self, pos):
        return pos[0] >= 0 and pos[0] < self.scene_x_dim and pos[1] >= 0 and pos[1] < self.scene_y_dim

    def _get_reward(self, state):
        r = np.zeros((self.reward_dimension, 1))
        r[state] = 1.0
        return r

    def play(self, action):
        pass

    @property
    def scene_x_dim(self):
        return self._size

    @property
    def scene_y_dim(self):
        return self._size


class MORLGridworld(Gridworld):
    """
    Multiobjective gridworld.
    """
    def __init__(self, size=10, gamma=0.9):
        self.gamma = gamma

        # self.actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
        self.actions = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        self.n_states = size * size
        self.n_states_print = self.n_states
        self._size = size
        self.reward_dimension = 3

        self.P = None
        self.R = None

        # Default Map as used in general MORL papers
        self._scene = np.zeros((size, size))
        self._scene[0, size-1] = 1
        self._scene[size-1, 0] = 1
        self._scene[size-1, size-1] = 1

        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    def name(self):
        return "MORL_Gridworld"

    def _get_reward(self, state):
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        if self._in_map(position) and self._scene[position] > 0:
            if state == 9:
                reward[0] = 1.0
            elif state == 90:
                reward[1] = 1.0
            elif state == 99:
                reward[2] = 1.0
        return reward

    def play(self, action):
        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]

        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
        else:
            self.last_state = state
            self.state = self._get_index(n_position)
            reward = self._get_reward(self.state)
            if (reward > 0).any():
                self.terminal_state = True

        return reward


class MORLGridworldTime(Gridworld):
    """
    Multiobjective gridworld.
    """
    def __init__(self, size=10, gamma=0.9):
        self.gamma = gamma

        # self.actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
        self.actions = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        self.n_states = size * size
        self.n_states_print = self.n_states
        self._size = size
        self.reward_dimension = 4

        self.P = None
        self.R = None

        # Default Map as used in general MORL papers
        self._scene = np.zeros((size, size))
        # self._scene[0, size-1] = 1
        self._scene[1, 7] = 1
        self._scene[size-1, 0] = 1
        self._scene[size-1, size-1] = 1

        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    def _get_reward(self, state):
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        if self._in_map(position) and self._scene[position] > 0:
            if state == 17:
                reward[0] = 1
            elif state == 90:
                reward[1] = 1
            elif state == 99:
                reward[2] = 1
        reward[-1] = -1
        return reward

    def play(self, action):
        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]

        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
        else:
            self.last_state = state
            self.state = self._get_index(n_position)
            reward = self._get_reward(self.state)
            if (reward > 0).any():
                self.terminal_state = True

        return reward


class MORLGridworldStatic(Gridworld):
    """
    Multiobjective gridworld.
    """
    def __init__(self, size=10, gamma=0.9):
        self.gamma = gamma

        self.actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        self.n_states = size * size
        self.n_states_print = self.n_states
        self._size = size
        self.reward_dimension = 4

        self.P = None
        self.R = None

        # Default Map as used in general MORL papers
        self._scene = np.zeros((size, size))
        self._scene[0, size-1] = 1
        self._scene[size-1, 0] = 1
        self._scene[size-1, size-1] = 1

        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    def _get_reward(self, state):
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        if self._in_map(position) and self._scene[position] > 0:
            if state == 9:
                reward[0] = 1
            elif state == 90:
                reward[1] = 1
            elif state == 99:
                reward[2] = 1
        reward[-1] = -0.1
        return reward

    def play(self, action):
        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]

        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
        else:
            self.last_state = state
            self.state = self._get_index(n_position)
            reward = self._get_reward(self.state)
            if (reward > 0).any():
                self.terminal_state = True

        return reward


class Financial(MORLProblem):
    """
    This is a multi-objective financial toy problem.
    The agent can choose to invest into a number of financial products. Each action means to buy
    or to sell one specific financial product. Therefore, if we have f financial products, there
    will be 2*f actions. The problem is formulated in a bandit-style setting. This means, we have
    only one state, in which we remain and can choose to buy or sell assets. The vector-valued
    reward contains the payouts for the current portfolio with respect to

    (payout, risk, flexibility)
    """

    def __init__(self, gamma=0.9):
        super(Financial, self).__init__(['state'])
        # TODO: financial environment not finished
        raise NotImplementedError("Financial environment not implemented yet.")

        self.P = None
        self.R = None

        self.n_states = 1
        self.n_financial_products = 3
        # buy and sell products or do nothing
        self.n_actions = 2 * self.n_financial_products + 1

        self.reward_dimension = 3

        self.product_rewards = [
            # Stock
            (1.0, -1.0, 1.0),
            # Bond
            (0.5, -0.1, -1.0),
            # CallMoney
            (0.2, 0.0, 0.0)
        ]

        self.reset()

        # build state transition matrix P_{ss'} where (i, j) is the transition probability
        # from state i to j
        if self.P is None:
            self._construct_p()

        # build reward vector R(s)
        if self.R is None:
            self._construct_r()

    @staticmethod
    def name(self):
        return "Financial"

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            for a in xrange(self.n_actions):
                for j in xrange(self.n_states):
                    self.P[i, a, j] = 1.0 / self.n_states

    def reset(self):
        self.state = 0
        self.last_state = 0
        self.terminal_state = False

    def _get_reward(self, state):
        # TODO: finish implementation not suitable for static LP solver
        r = np.zeros((self.reward_dimension, 1))
        r[0] = 1.0
        return r

    def play(self, action):
        """
        Perform an action (buy or sell an asset)

        Parameters
        ----------
        action: integer, which asset will be bought/sold by
            the agent.


        Returns
        -------
        reward: reward of the current state.
        """

        return self._get_reward(self.state)


class MORLRobotActionPlanning(MORLProblem):
    """
    This is supposed to be a example problem for a high level task selection
    in a robot, which can receive vector valued reward.
    """

    def __init__(self, gamma=0.9):
        super(MORLRobotActionPlanning, self).__init__(['state'])

        self.P = None
        self.R = None
        self.gamma = gamma

        # States:
        # 	s0 - standby
        #   s1 - exploration
        #   s2 - gathering
        #   s3 - recharge
        #   s4 - self-repair
        self.n_states = 5
        self.init_state = 0
        # Actions:
        #   Standby:        [0, 0, 0, 0.1]
        #   Exploration:    [1, 0, 0.5, 0.5]
        #   Gathering:      [0, 1, 1, 1]
        #   Recharge:       [0, 0, 0.1, -1]
        #   Self-Repair:    [0, 0, -1, 0.2]
        self.n_actions = 5

        # Rewards:
        #   (Information, Utility of Resources, Wear, Energy Consumption)
        self.reward_dimension = 4

        self.reset()

        # build state transition matrix P_{ss'} where (i, j) is the transition probability
        # from state i to j
        if self.P is None:
            self._construct_p()

        # build reward vector R(s)
        if self.R is None:
            self._construct_r()

    @staticmethod
    def name(self):
        return "RobotActionPlaning"

    def _construct_p(self):
        #   Standby:        [0, 0, 0, 0.1]
        #   Exploration:    [1, 0, 0.5, 0.5]
        #   Gathering:      [0, 1, 1, 1]
        #   Recharge:       [0, 0, 0.1, -1]
        #   Self-Repair:    [0, 0, -1, 0.2]
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            for a in xrange(self.n_actions):
                # always transit to the state given by the action
                self.P[i, a, a] = 1.0

    def reset(self):
        self.state = 0
        self.last_state = 0
        self.terminal_state = False

    def _get_reward(self, state):
        r = np.zeros((self.reward_dimension, 1))
        #   Standby:        [0, 0, 0, 0.1]
        #   Exploration:    [1, 0, 0.5, 0.5]
        #   Gathering:      [0, 1, 1, 1]
        #   Recharge:       [0, 0, 0.1, -1]
        #   Self-Repair:    [0, 0, -1, 0.2]
        if state == 0:
            # Standby
            r = np.array([0, 0, 0, 0.1])
        elif state == 1:
            # Exploration
            r = np.array([1, 0, 0.5, 0.5])
        elif state == 2:
            # Gathering
            r = np.array([0, 1, 1, 1])
        elif state == 3:
            # Recharge
            r = np.array([0, 0, 0.1, -1])
        elif state == 4:
            # Self-Repair
            r = np.array([0, 0, -1, 0.2])
        else:
            raise RuntimeError("Unknown state in reward encountered")
        return r

    def play(self, action):
        """
        Perform an action

        Parameters
        ----------
        action: integer, what high-level decision should be taken

        Returns
        -------
        reward: reward of the current state.
        """
        self.last_state = self.state
        self.state = sampleFromDiscreteDistribution(1,
                                                    self.P[self.last_state,
                                                           action, :])
        return self._get_reward(self.state)


class MORLBuridansAss1DProblem(MORLProblem):
    """
    This problem contains buridans ass domain. An ass starts (usually) in a 3x3 grid in the middle position (1,1)
    in the top left and the bottom right corner there is a pile of food. if the ass moves away from a visible foodstate,
    the food in the bigger distance will be stolen with a probability of p. Eeating the food means choosing action
    "stay" at the field of a food
    it will be rewarded with following criteria: hunger, lost food, walking distance
    hunger means
    """

    def __init__(self, size=3, p=0.9, n_appear=10, gamma=0.9):

        self.steal_probability = p
        # available actions: stay                right,             up,                 left,            down
        self.actions = (np.array([0, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        # steps until new food is generated
        self.n_appear = n_appear
        self.gamma = gamma
        # size of the grid
        self.n_states = size * size
        self.n_states_print = self.n_states
        # size of the grid in one dimension
        self._size = size
        # dimensions: 0: hunger(time, the ass hasn't got eaten(-1 per t)), 1: lost food(-0.5), 2: distance walked(-1)
        self.reward_dimension = 3
        # food positions
        self.food1 = self._size-1, 0
        self.food2 = 0, self._size-1
        # scene quadradic zeros
        self._scene = np.zeros((self._size, self._size))
        # at places where the food is: 1
        self._scene[0, 0] = 1
        self._scene[self._size-1, self._size-1] = 1

        # pythagoras distance
        self.max_distance = sqrt(2)

        # initial state is the middle (e.g. at 3x3 matrix index 4)
        init = (self._size*self._size)/2
        self.state = init
        self.last_state = init
        self.terminal_state = False
        # counting variable for food recreation
        self.count = 0
        # counting variable for hunger
        self.hunger = 0
        self._flat_map = np.argwhere(self._scene >= 0)  # get all indices greater than zero
        # get all elements greater than zero and stack them to the corresponding index
        self._flat_map = np.column_stack((self._flat_map,  self._scene[self._scene >= 0]))
        self.P = None
        self._construct_p()
        self.R = None
        self._construct_r()

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            for a in xrange(self.n_actions):
                for j in xrange(self.n_states):
                    if not self._in_map(self._get_position(j)):
                        self.P[i, a, i] = 1.0
                    else:
                        self.P[i, a, j] = 1.0

    def reset(self):
        init = (self._size*self._size)/2
        self.state = init
        self.last_state = init
        self.terminal_state = False
        self.count = 0
        self.hunger = 0
        self._scene[self._size-1, self._size-1] = 1
        self._scene[0, 0] = 1

    def name(self):
        return "BuridansAss"

    def _get_reward(self, state):
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        # check if food is visibly reachable:
        # if not and in self.steal_probability cases the food is stolen
        if self._get_distance(position, self.food1) > self.max_distance and random.random() <\
                self.steal_probability:
            self._scene[self.food1] = 0
            reward[1] = -0.5
        # same for food no. 2
        if self._get_distance(position, self.food2) > self.max_distance and random.random() <\
                self.steal_probability:
            self._scene[self.food2] = 0
            reward[1] = -0.5
        # check if we're eating something and reward, finally resetting hunger
        if self._in_map(position) and self._scene[position] > 0 and self.last_state == self.state:
            reward[0] = 1
            self.hunger = 0
        else:
            # negative reward if we're walking to much without food
            self.hunger += 1
            if self.hunger > 9:
                reward[0] = -1
        # check if we're walking. if positive, reward: -1
        if self.last_state != self.state:
            reward[2] = -1

        return reward

    def play(self, action):
        # count actions
        self.count += 1
        # after 10 steps eventually stolen food is reproduced
        if self.count == self.n_appear:
            self._scene[self._size-1, self._size-1] = 1
            self._scene[0, 0] = 1

        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]
        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
        else:

            self.last_state = state
            self.state = self._get_index(n_position)
            reward = self._get_reward(self.state)
            if (reward > 0).any():
                self.terminal_state = True
        return reward

    def _get_distance(self, state1, state2):
        first = np.array([state1[0], state1[1]])
        second = np.array([state2[0], state2[1]])
        return np.linalg.norm(second-first)

    def _get_index(self, position):
        return position[0] * self.scene_x_dim + position[1]

    def _get_position(self, index):
        return index // self.scene_y_dim, index % self.scene_x_dim

    def _in_map(self, pos):
        return pos[0] >= 0 and pos[0] < self.scene_x_dim and pos[1] >= 0 and pos[1] < self.scene_y_dim

    @property
    def scene_x_dim(self):
        return self._size

    @property
    def scene_y_dim(self):
        return self._size

    def print_map(self, pos=None):
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()


class MORLBuridansAssProblem(MORLProblem):
    """
    This problem contains buridans ass domain. An ass starts (usually) in a 3x3 grid in the middle position (1,1)
    in the top left and the bottom right corner there is a pile of food. if the ass moves away from a visible foodstate,
    the food in the bigger distance will be stolen with a probability of p. Eeating the food means choosing action
    "stay" at the field of a food
    it will be rewarded with following criteria: hunger, lost food, walking distance
    hunger means that after 9 steps without eating a food pile the next action(s) will be rewarded with -1
    """

    def __init__(self, size=3, p=0.9, n_appear=10, gamma=1.0):
        self.steal_probability = p
        # available actions: stay                right,             up,                 left,            down
        self.actions = (np.array([0, 0]),  np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        # steps until new food is generated
        self.n_appear = n_appear
        self.gamma = gamma
        # size of the grid times 9 time states for hunger, 4 food states
        self.n_states = size * size * 9 * 4
        # visible is only the 3x3 grid
        self.n_states_print = size*size
        # size of the grid in one dimension
        self._size = size
        # dimensions: 0: hunger(if he eats: +1), 1: lost food(-0.5), 2: distance walked(-1)
        self.reward_dimension = 3
        # maximal step count without eating
        self.max_hunger = 9
        # food positions
        self.food1 = (0, 0)
        self.food2 = (self._size-1, self._size-1)
        self.food = [1, 1]
        # scene quadradic zeros
        self._scene = np.zeros((self._size, self._size))
        # at places where the food is: 1
        self._scene[0, 0] = 1
        self._scene[self._size-1, self._size-1] = 1

        # pythagoras distance
        self.max_distance = sqrt(2)
        # some dicts to map the states in indices and reverse:#
        #######################################################
        # map cartesian coordinates in indices
        self.position_map = dict()
        for y in xrange(self.scene_y_dim):
            for x in xrange(self.scene_x_dim):
                self.position_map[x, y] = len(self.position_map)
        # map four states of food places in indices:
        self.food_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        self.rev_food_map = {0: [0, 0], 1: [0, 1], 2: [1,0], 3: [1, 1]}
        # for all states
        self.state_map = dict()
        # just for 2d Plotting:
        self.plot_map = dict()
        for y in xrange(self.scene_y_dim):
            for x in xrange(self.scene_x_dim):
                self.plot_map[x, y] = len(self.plot_map)
                for f in xrange(len(self.food_map)):
                    for h in xrange(self.max_hunger):
                        self.state_map[self.position_map[x, y], f, h] = len(self.state_map)
        # initial state is the middle (e.g. at 3x3 matrix index 4), both food piles, not hungry
        self.init_state = self.state_map[self.position_map[1, 1], 3, 0]
        self.state = self.init_state
        self.last_state = self.init_state
        self.terminal_state = False
        # counting variable for food recreation
        self.count = 0
        # counting variable for hunger
        self.hunger = 0
        self._flat_map = np.argwhere(self._scene >= 0)  # get all indices greater than zero
        # get all elements greater than zero and stack them to the corresponding index
        self._flat_map = np.column_stack((self._flat_map,  self._scene[self._scene >= 0]))
        self.P = None
        self._construct_p()
        self.R = None
        self._construct_r()

    def create_plottable_states(self, states):
        pos = [[self._get_position(s) for s in states[i]] for i in xrange(len(states))]
        plt_states = [[self.position_map[p[0], p[1]] for p in pos[l]] for l in xrange(len(pos))]
        ret_states = [np.array(plt_states[i]) for i in xrange(len(plt_states))]
        ret_states = np.array(ret_states)
        return ret_states

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            pos, food, hunger = self._get_position(i, complete=True)
            f1, f2 = self.rev_food_map[food][0], self.rev_food_map[food][1]
            y, x = self.get_cartesian_coordinates_from_pos_state(pos)
            for a in xrange(self.n_actions):
                ax, ay = self.actions[a]
                nx, ny = x+ax, y+ay
                if not self._in_map((ny, nx)):
                    new_state = self.state_map[pos, food, new_hunger]
                    self.P[i, a, new_state] = 1.0
                    continue
                n_pos_s = self.position_map[ny, nx]

                if a != 0:
                    new_hunger = hunger+1 if(hunger >= self.max_hunger) else self.max_hunger-1
                else:
                    new_hunger = hunger

                # actions away from a food will probably cause a transition in another food state, if it looses fight
                if (y, x) == (0, 0) or (y, x) == (self._size, 0) or (y, x) == (0, self._size):
                    if self._get_distance((ny, nx), self.food1) > self.max_distance:
                        new_f_st = self.food_map[0, f2]
                        new_s = self.state_map[n_pos_s, new_f_st, new_hunger]
                        self.P[i, a, new_s] = self.steal_probability
                        new_s = self.state_map[n_pos_s, food, new_hunger]
                        self.P[i, a, new_s] = 1 - self.steal_probability
                    if self._get_distance((ny, nx), self.food2) > self.max_distance:
                        new_f_st = self.food_map[f1, 0]
                        new_s = self.state_map[n_pos_s, new_f_st, new_hunger]
                        self.P[i, a, new_s] = self.steal_probability
                        new_s = self.state_map[n_pos_s, food, new_hunger]
                        self.P[i, a, new_s] = 1 - self.steal_probability
                if (y, x) == self.food1:
                    if a == 0:
                        new_f_st = self.food_map[0, f2]
                        new_s = self.state_map[pos, new_f_st, 0]
                        self.P[i, a, new_s]
                if (y, x) == self.food2:
                    if a == 0:
                        new_f_st = self.food_map[f1, 0]
                        new_s = self.state_map[pos, new_f_st, 0]
                        self.P[i, a, new_s]
                else:
                    new_state = self.state_map[n_pos_s, food, new_hunger]
                    self.P[i, a, new_state] = 1.0

    def get_all_states_with_that_pos(self, pos):
        pass

    def get_cartesian_coordinates_from_pos_state(self, state):
        for o in self.position_map.keys():
            if self.position_map[o] == state:
                cart_pos = o
        return cart_pos[0], cart_pos[1]

    def reset(self):
        self.state = self.init_state
        self.last_state = self.init_state
        self.terminal_state = False
        self.count = 0
        self.hunger = 0
        self._scene[self._size-1, self._size-1] = 1
        self._scene[0, 0] = 1

    def name(self):
        return "BuridansAss"

    def _get_reward(self, state):
        # the reward function should return a reward only by knowing the state
        pos, food, hunger = self._get_position(state, complete=True)
        pos = self._get_position(state, complete=False)
        reward = np.zeros(self.reward_dimension)
        # if we've eaten something, hunger state is 0 and some food is missing on the field
        if hunger == 0:
            # check if we're on first food field and if this food is missing
            if (pos == self.food1) and (food == 0 or food == 1):
                reward[0] = 1
            # check if we're one second food field and if this food is missing
            if (pos == self.food2) and (food == 0 or food == 2):
                reward[0] = 1
        # if hunger state is to big (>8), penalty:
        if hunger == self.max_hunger-1:
                reward[2] = -1
        # if food state is not complete full and hunger is not zero, a food pile must have been stolen
        if hunger > 0 and food != self.food_map[1, 1] and self.count == 0:
            reward[1] = -0.5
        return reward

    def play(self, action):
        # count actions
        self.count += 1
        # after 10 steps eventually stolen food is reproduced
        if self.count == self.n_appear:
            self._scene[self._size-1, self._size-1] = 1
            self._scene[0, 0] = 1
            self.food = [1, 1]
            self.count = 0
        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]
        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
            if (reward > 0).any():
                self.terminal_state = True
            return reward
        # hunger grows if we're going
        if action != 0:
            self.hunger += 1
            if self.hunger > self.max_hunger-1:
                self.hunger = self.max_hunger-1
        if self._scene[n_position[0], n_position[1]] > 0 and action == 0:
            self.hunger = 0

            if (n_position == self.food1).all():
                self.food[0] = 0
            else:
                self.food[1] = 0
        # maybe food is stolen if we go in the wrong direction
        elif self._get_distance(n_position, self.food1) > self.max_distance and random.random() <\
                self.steal_probability and self.food[0]:
            self._scene[self.food1] = 0
            self.count = 0
            self.food[0] = 0
        # same for food no. 2
        elif self._get_distance(n_position, self.food2) > self.max_distance and random.random() <\
                self.steal_probability and self.food[1]:
            self._scene[self.food2] = 0
            self.count = 0
            self.food[1] = 0
        self.last_state = state
        self.state = self._get_index(n_position)
        reward = self._get_reward(self.state)
        if (reward > 0).any():
            self.terminal_state = True
        return reward

    def _get_distance(self, state1, state2):
        first = np.array([state1[0], state1[1]])
        second = np.array([state2[0], state2[1]])
        return np.linalg.norm(second-first)

    def _get_index(self, position):
        x = sys._getframe(1)
        if x.f_code.co_name == '_policy_plot2':
            return position[0] * self.scene_x_dim + position[1]
        pos = self.position_map[position[1], position[0]]
        f = self.food_map[self.food[0], self.food[1]]
        h = self.hunger
        return self.state_map[pos, f, h]

    def _get_position(self, index, complete=False):
        for k in self.state_map.keys():
            if self.state_map[k] == index:
                pos = k
                break
        for o in self.position_map.keys():
            if self.position_map[o] == pos[0]:
                cart_pos = o
                break
        if complete:
            return pos[0], pos[1], pos[2]

        return cart_pos[1], cart_pos[0]

    def _in_map(self, pos):
        return pos[0] >= 0 and pos[0] < self.scene_y_dim and pos[1] >= 0 and pos[1] < self.scene_x_dim

    @property
    def scene_x_dim(self):
        return self._size

    @property
    def scene_y_dim(self):
        return self._size

    def print_map(self, pos=None):
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()


class MOPuddleworldProblem(MORLProblem):
    """
    This problem contains a quadratic map (please use size more than 15, to get a useful puddle)
    the puddle is an obstacle that the agent has to drive around. The aim is to reach the goal state at the top right
    """
    def __init__(self, size=20, gamma=0.9):

        # available actions:    right,             down,                 left,            up
        self.actions = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        self.gamma = gamma
        # size of the grid
        self.n_states = size * size
        self.n_states_print = self.n_states
        # size of the grid in one dimension
        self._size = size
        # dimensions: 0:goal (not) reached, 1: puddle touched(-1/-2)
        self.reward_dimension = 2
        # goal position
        self.goal = [0, self._size-1]
        # scene quadradic zeros
        self._scene = np.zeros((self._size, self._size))
        # create puddle: the deeper, the greater the regret
        if self._size != 20:
            self._scene[0.1*self._size:0.7*self._size, 0.09*self._size:0.5*self._size+1] = -2.0
            self._scene[0.35 * self._size:, :0.30 * self._size] = 0.0
            self._scene[0.10 * self._size, :0.3 * self._size] = 0.0
            self._scene[0.10*self._size, (self._size/2)] = 0.0
            self._scene[0.35*self._size:, (self._size/2)] = 0.0
            self._scene[0.2*self._size:0.3*self._size, 0.14*self._size:0.5*self._size] = -4.0
            self._scene[0.15*self._size:0.65*self._size, 0.35*self._size:0.45*self._size] = -4.0
            self._scene[2:14, 1:10] = -4.0
            self._scene[0, self._size-1] = 1.0
        else:
            self._scene[2:14, 2:11] = -2.0
            self._scene[7:, :6] = 0.0

            self._scene[2, :6] = 0.0

            self._scene[2, 10] = 0.0
            self._scene[7:, 10] = 0.0
            self._scene[4:6, 2:10] = -4.0
            self._scene[3:13, 7:9] = -4.0
            self._scene[0, 19] = 1.0

        self.state_map = dict()
        for y in xrange(self._size):
            for x in xrange(self._size):
                self.state_map[y, x] = len(self.state_map)

        # all possible states
        self.init = [i for i in xrange(self._size * self._size - 1)]
        # we don't wanna start in goal state and right next to it, so we delete these states
        del self.init[self._size - 1]
        del self.init[self._size - 2]
        del self.init[(2 * self._size) - 1]
        # initial state is randomly selected (non-goal)
        self.init_state = random.choice(self.init)
        self.state = self.init_state
        self.last_state = self.init_state
        self.terminal_state = False
        # # plot
        # self.fig, self.ax = plt.subplots()
        # temp = self._scene
        # # self.ax.imshow(temp, interpolation='nearest')
        # step = 1.
        # min = 0.
        # rows = temp.shape[0]
        # columns = temp.shape[1]
        # row_arr = np.arange(min, rows)
        # col_arr = np.arange(min, columns)
        # x, y = np.meshgrid(row_arr, col_arr)
        # for col_val, row_val in zip(x.flatten(), y.flatten()):
        #     c = int(temp[row_val, col_val])
        #     self.ax.text(col_val, row_val, c, va='center', ha='center')
        # self._flat_map = np.argwhere(self._scene >= 0)  # get all indices greater than zero
        # # get all elements greater than zero and stack them to the corresponding index
        # self._flat_map = np.column_stack((self._flat_map,  self._scene[self._scene >= 0]))
        # # plt.show()
        self.P = None
        # self._construct_p()
        self.R = None
        self._construct_r()

    def _set_new_init(self):
        self.init_state = random.choice(self.init)

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            for a in xrange(self.n_actions):
                for j in xrange(self.n_states):
                    if not self._in_map(self._get_position(j)):
                        self.P[i, a, i] = 1.0
                    else:
                        self.P[i, a, j] = 1.0

    def reset(self):
        self._set_new_init()
        self.state = self.init_state
        self.last_state = self.init_state
        self.terminal_state = False

    def _get_reward(self, state):
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        if self._in_map(position) and self._scene[position] < 0:
            reward[1] = self._scene[position]*10
        if state == self._size:
            reward[0] = 1
        else:
            reward[0] = -1

        return reward

    def name(self):
        return "Puddleworld"

    def play(self, action):
        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]
        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
        else:
            self.last_state = state
            self.state = self._get_index(n_position)
            reward = self._get_reward(self.state)
            if (reward > 0).any():
                self.terminal_state = True
        # self.print_map(position)
        return reward

    def _get_index(self, position):
        y, x = position[0], position[1]
        return self.state_map[y, x]

    def _get_position(self, index):
        return index // self.scene_x_dim, index % self.scene_x_dim

    def _in_map(self, pos):
        return pos[0] >= 0 and pos[0] < self.scene_y_dim and pos[1] >= 0 and pos[1] < self.scene_x_dim

    @property
    def scene_x_dim(self):
        return self._size

    @property
    def scene_y_dim(self):
        return self._size

    def print_map(self, pos=None):
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()


class MORLResourceGatheringProblem(MORLProblem):
    """
    In this problem the agent has to find the resources and bring them back to the homebase.
    the enemies steal resources with a probability of 0.9

    """
    def __init__(self, size=5, gamma=0.9, p=0.9):
        self.multidimensional_states = True
        # for each field in the scene we have 4 possible states, each shows the state of the bag in one position
        self._bag_mapping = {
            0: [0, 0],
            1: [0, 1],
            2: [1, 0],
            3: [1, 1]
        }
        self.position_map = dict()
        for y in xrange(size):
            for x in xrange(size):
                self.position_map[x, y] = len(self.position_map)
        # available actions:    right,             up,                 left,            down
        self.actions = (np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0]))
        # how many actions ?
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        # gamma is learning rate
        self.gamma = gamma
        # the probability, the bag will be rubbed, in case of enemy attack
        self.losing_probability = p
        # the bag can contain two elements and is empy at the beginning
        self._bag = [0, ] * 2
        # we have a grid of 5 columns and 5 rows, for each cartesian state we have 2 states for
        # each of two bag places (--> 4D) a place in the bag can be 0 or 1 (empty or filled)
        self.n_bag_place_states = 2
        self.n_states = size * size * self.n_bag_place_states * self.n_bag_place_states
        self.n_states_print = size*size

        # size of the grid in one dimension
        self._size = size
        # dimensions: 0:attack of enemy(-1), 1: resource 1 (+1), 2: resource 2 (+1)
        self.reward_dimension = 3
        # scene quadradic zeros, the game field in cartesian
        self._scene = np.zeros((self._size, self._size))
        # enemies coordinates get negative field
        self.enemy_positions = [(0, 3), (1, 2)]
        self._scene[0, 3] = -1
        self._scene[1, 2] = -1
        # resources (are on state 8 and 9):
        self.resource_positions = [(0, 2), (1, 4)]
        self._scene[0, 2] = 1
        self._scene[1, 4] = 1
        # init state (homebase)
        self.init = 88
        self.init_position = [4, 2]
        self.state = self.init
        self.last_state = self.init
        self.terminal_state = False
        self.P = None
        self._construct_p()
        self.R = np.zeros((self.n_states, self.reward_dimension))
        self._construct_r()

    def get_bag_index(self, bag):
        for i in self._bag_mapping.keys():
            if self._bag_mapping[i] == bag:
                return i

    def reset(self):
        self.init = 88
        self.state = self.init
        self.last_state = self.init
        self.terminal_state = False
        self._scene[0, 2] = 1
        self._scene[1, 4] = 1

    def create_plottable_states(self, states):
        pos = [[self._get_position(s) for s in states[i]] for i in xrange(len(states))]
        plt_states = [[self.position_map[p[0], p[1]] for p in pos[l]] for l in xrange(len(pos))]
        ret_states = [np.array(plt_states[i]) for i in xrange(len(plt_states))]
        ret_states = np.array(ret_states)
        return ret_states

    def _get_reward(self, state):
        pos = self._get_position(state)
        position = pos[0], pos[1]
        bag = pos[2:]
        reward = np.zeros(self.reward_dimension)
        # we are 1. in the map, 2. we're on a field where the enemy is and 3. we lost the fight
        if self._scene[position] < 0 and random.random() < self.losing_probability:
            # we  get negative reward
            reward[0] = -1
            # we need to get back to the homebase
            return reward

        # if we're turning back home
        if position == (4, 2):
            # we get reward for the bag (0,1,0), (0,1,1), (0,0,1), or (0,0,0)
            reward[1:] = bag
        self.state = self._get_index((position[0], position[1], self._bag[0], self._bag[1]))
        return reward

    def _construct_p(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            xi, yi, bag1, bag2 = self._get_position(i)
            for a in xrange(self.n_actions):
                prob = 1.0
                ox, oy = self.actions[a]
                tx, ty = xi + ox, yi + oy
                bag_state = bag1, bag2

                if (tx, ty) in self.resource_positions:
                    tb0 = 1 if (tx, ty) == self.resource_positions[0] and not bag_state[0] else 0
                    tb1 = 1 if (tx, ty) == self.resource_positions[1] and not bag_state[1] else 0
                else:
                    tb0, tb1 = bag_state
                if (tx, ty) in self.enemy_positions and bag_state != (0, 0):
                    tb0, tb1 = 0, 0
                    prob = self.losing_probability
                    n = self._get_index((tx, ty, tb0, tb1))
                    self.P[i, a, n] = prob
                    n = self._get_index((tx, ty, bag1, bag2))
                    self.P[i, a, n] = 1-prob
                    continue
                n = self._get_index((tx, ty, tb0, tb1))
                if not self._in_map((tx, ty, tb0, tb1)):
                    self.P[i, a, i] = prob
                else:
                    self.P[i, a, n] = prob

    def name(self):
        return "Resource Gathering"

    def _construct_r(self):
        self.R = np.zeros((self.n_states, self.reward_dimension))
        for s in xrange(self.n_states):
            for a in xrange(len(self.actions)):
                xi, yi, bag1, bag2 = self._get_position(s)
                # we get reward if we come home with non-empty bag
                if [xi, yi] == self.init_position:
                    self.R[s, 1] = bag1
                    self.R[s, 2] = bag2
                if [xi, yi] == self.enemy_positions[0] or [xi, yi] == self.enemy_positions[1]:
                    if bag1 or bag2:
                        self.R[s, 0] = -1

    def play(self, action):
        actions = self.actions
        state = self.state

        position = self._get_position(state)[:2]
        bag = self._get_position(state)[2:]
        n_position = position + actions[action]
        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)

        if self._in_map(position) and self._scene[position] < 0 and random.random() < self.losing_probability:
            # our resources is stolen
            self._bag[:] = [0, ] * len(self._bag)
            # we need to get back to the homebase
            position = self.init_position
            self.state = self._get_index((position[0], position[1], self._bag[0], self._bag[1]))
            reward = self._get_reward(self.state)
            return reward
        # we are 1. in the map and 2. we found resource:
        elif self._in_map(position) and self._scene[position] > 0:  # put the resource in our bag
            # check which resource it is
            if position == self.resource_positions[0]:
                self._bag[0] = 1
            if position == self.resource_positions[1]:
                self._bag[1] = 1
            # delete it from the field
            self._scene[position] = 0

        # if we're turning back home
        if self._in_map(n_position) and (n_position == (4, 2)).all():
            # we get reward for the bag (0,1,0), (0,1,1), (0,0,1), or (0,0,0)
            reward[1:] = self._bag
        self.state = self._get_index((n_position[0], n_position[1], self._bag[0], self._bag[1]))
        self.last_state = state

        reward = self._get_reward(self.state)
        if (reward > 0).any():
            self.terminal_state = True
        return reward

    def _get_index(self, position):
        x = sys._getframe(1)
        if x.f_code.co_name == '_policy_plot2':
            return position[0] * self.scene_x_dim + position[1]
        bagind = self.get_bag_index([position[2], position[3]])
        return (position[0] * self.scene_x_dim + position[1])*len(self._bag_mapping) + bagind

    def _get_position(self, index):
        bag_state = index % len(self._bag_mapping)
        bag = self._bag_mapping[bag_state]
        index /= len(self._bag_mapping)
        return index // self.scene_x_dim, index % self.scene_y_dim, bag[0], bag[1]

    def _in_map(self, pos):
        return pos[0] >= 0 and pos[0] < self.scene_y_dim and pos[1] >= 0 and pos[1] < self.scene_x_dim

    @property
    def scene_x_dim(self):
        return self._size

    @property
    def scene_y_dim(self):
        return self._size

    def print_map(self, pos=None):
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()

