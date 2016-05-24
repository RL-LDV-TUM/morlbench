#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
@author: Johannes Feldmaier <johannes.feldmaier@tum.de>

"""

from helpers import SaveableObject, loadMatrixIfExists, virtualFunction
from probability_helpers import assureProbabilityMatrix

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sqrt
import logging as log
import random
import time

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

        self._flat_map = np.argwhere(self._scene>=0)  # get all indices greater than zero
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

                        if self._in_map(n_pos) and self._flat_map[pos_index, 2] == 0 and n_pos_index >= 0:  # we are in the map and no special state
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

    def __str__(self):
        return self.__class__.__name__

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
    def __init__(self, state=-0.5, gamma=0.9):
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

        # Discount Factor
        self.gamma = gamma

        self._minPosition = -1.2  # Minimum car position
        self._maxPosition = 0.6  # Maximum car position (past goal)
        self._maxVelocity = 0.07  # Maximum velocity of car
        self._goalPosition = 0.5  # Goal position - how to tell we are done

        self._accelerationFactor = 0.001
        self._maxGoalVelocity = 0.07

        self._start_state = state
        self._velocity = 0
        self.state = self._start_state
        self.last_state = self._start_state
        self._position = self.get_position(self._start_state)

        self._time = 0

        self._default_reward = 1

        self.terminal_state = False

        self.n_states = 100  # TODO: Discretize Mountain car states!
        self.n_states_print = self.n_states - 1
        self.reward_dimension = 2

    def reset(self):
        self._velocity = 0
        self.state = self._start_state
        self.last_state = self._start_state
        self._position = self.get_position(self._start_state)

    def get_position(self, state):
        return state

    def get_state(self, position):
        pass

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
        previousState = self._state

        self._time += 1

        map_actions = {
            'none': 0,  # coasting
            'right': 1,  # forward thrust
            'left': -1,  # backward thrust
        }

        # Determine acceleration factor
        if action < len(self.actions):
            factor = map_actions[self.actions[action]]  # map action to thrust factor
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0  # Default action

        self.car_sim(factor)

        if self.terminal_state:
            return [self._default_reward, self._time]
        else:
            return [0, self._time]

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
        velocity_change = self._accelerationFactor * factor - 0.0025 * cos(3 * self._position)

        self._velocity = minmax(self._velocity + velocity_change, -self._maxVelocity, self._maxVelocity)

        self._position += self._velocity

        self._position = minmax(self._position, self._minPosition, self._maxPosition)

        if self._position <= self._minPosition:  # and (self._velocity < 0)
            self._velocity = 0.0

            # if self._position >= self._goalPosition and abs(self._velocity) > self._maxGoalVelocity:
            #    self._velocity = -self._velocity

            # TODO: set terminal state for being at the goal position


class MountainCarMulti(MountainCar):
    def __init__(self, state=0.5):
        """
        Initialize the Multi Objective Mountain car problem.

        Parameters
        ----------
        state: default state is -0.5
        """
        self._nb_actions = 0

        super(MountainCarMulti, self).__init__(state=state)

        self.reward_dimension = 3

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
        previousState = self.state

        self._time += 1

        map_actions = {
            'left': -1,  # backward thrust
            'right': 1,  # forward thrust
            'none': 0,  # coasting
        }

        # Determine acceleration factor
        if action < len(self.actions):
            factor = map_actions[self.actions[action]]  # map action to thrust factor
            if (self._actions[action] == 'right') or (self.actions[action] == 'left'):
                self._nb_actions += 1
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0  # Default action

        self.car_sim(factor)

        if self.terminal_state:
            return [self._default_reward, self._time]
        else:
            return [0, self._time]


class Gridworld(MORLProblem):
    """
    Original Algen-Gridworld.
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
        return index // self.scene_y_dim, index % self.scene_x_dim

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


class MORLBurdiansAssProblem(MORLProblem):
    """
    This problem contains buridans ass domain. An ass starts (usually) in a 3x3 grid in the middle position (1,1)
    in the top left and the bottom right corner there is a pile of food. if the ass moves away from a visible food state,
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

        # initial state is the middle (e.g. at 3x3 matrix index 4)
        init = (self._size*self._size)/2
        self.state = init
        self.last_state = init
        self.terminal_state = False
        # pythagoras distance
        self.max_distance = sqrt(2)
        # counting variable for food recreation
        self.count = 0
        # counting variable for hunger
        self.hunger = 0

    def reset(self):
        init = (self._size*self._size)/2
        self.state = init
        self.last_state = init
        self.terminal_state = False
        self.count = 0
        self.hunger = 0

    def _get_reward(self, state):
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        # check if food is visibly reachable:
        # if not and in self.steal_probability cases the food is stolen
        if self._get_distance(position, self.food1) > self.max_distance and random.random() <\
                self.steal_probability:
            self._scene[self.food1] = 0
            reward[1] -= 0.5
        # same for food no. 2
        if self._get_distance(position, self.food2) > self.max_distance and random.random() <\
                self.steal_probability:
            self._scene[self.food2] = 0
            reward[1] -= 0.5
        # check if we're eating something and reward, finally resetting hunger
        if self._in_map(position) and self._scene[position] > 0 and self.last_state == self.state:
            reward[0] = 1
            self.hunger = 0
        else:
            # negative reward if we're walking to much without food
            self.hunger += 1
            if self.hunger > 8:
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


class MOPuddleworldProblem(MORLProblem):
    """
    This problem contains a quadratic map (please use size more than 15, to get a useful puddle)
    the puddle is an obstacle that the agent has to drive around. The aim is to reach the goal state at the top right
    """
    def __init__(self, size=20, gamma=0.9):

        # available actions:    right,             up,                 left,            down
        self.actions = (np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.array([1, 0]))
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
        self._scene[0.1*self._size:0.7*self._size, 0.05*self._size:0.5*self._size+1] = -1
        self._scene[0.35*self._size:, :0.30*self._size] = 0
        self._scene[0.10*self._size, :0.3*self._size] = 0
        self._scene[0.10*self._size, (self._size/2)] = 0
        self._scene[0.35*self._size:, (self._size/2)] = 0
        self._scene[0.2*self._size:0.3*self._size, 0.1*self._size:0.5*self._size] = -2
        self._scene[0.15*self._size:0.65*self._size, 0.35*self._size:0.45*self._size] = -2
        self._scene[0, self._size-1] = 1

        # all possible states
        self.intstates = [i for i in xrange(self._size*self._size-1)]
        # we don't wanna start in goal state
        del self.intstates[self._size-1]
        # initial state is randomly selected (non-goal)
        init = random.choice(self.intstates)
        self.state = init
        self.last_state = init
        self.terminal_state = False
        # plot
        self.fig, self.ax = plt.subplots()
        temp = self._scene

        self.ax.imshow(temp, interpolation='nearest')
        step = 1.
        min = 0.
        rows = temp.shape[0]
        columns = temp.shape[1]
        row_arr = np.arange(min, rows)
        col_arr = np.arange(min, columns)
        x, y = np.meshgrid(row_arr, col_arr)
        for col_val, row_val in zip(x.flatten(), y.flatten()):
            c = int(temp[row_val, col_val])
            self.ax.text(col_val, row_val, c, va='center', ha='center')
        plt.show()

    def reset(self):
        init = random.choice(self.intstates)
        self.state = init
        self.last_state = init
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

class MORLResourceGatheringProblem:

    def __init__(self):
        pass

