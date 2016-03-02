#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import SaveableObject, loadMatrixIfExists, virtualFunction
from probability_helpers import assureProbabilityMatrix

import numpy as np
import matplotlib.pyplot as plt
from math import cos
import logging as log
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

    def play(self):
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
            actions = (np.array([-1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, -1]), np.array([0, 0]))
            # actions = (np.array([-1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, -1]))

        if scene is None:
            # Default Map as used in general MORL papers
            self._scene = np.zeros((11, 10))
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
            self.P = loadMatrixIfExists(os.path.join('defaults', str(self) + '_default_P.pickle'))
            self.R = loadMatrixIfExists(os.path.join('defaults', str(self) + '_default_R.pickle'))

        self._flat_map = np.ravel(self._scene, order='C')  # flat map with C-style order (column-first)

        self.n_states = (self._scene.shape[0] * self._scene.shape[1]) + 1 # +1 for terminal state
        self._index_terminal_state = self.n_states - 1

        #self._predictor_accuracy = predictor_accuracy
        #self._payouts = payouts
        self.actions = actions
        self.n_actions = len(self._actions)

        self.reward_dimension = 2
        self._extended_reward = extended_reward
        if extended_reward:
            # self._reward_dimension += self.n_states
            self.reward_dimension = self.n_states

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
                valid_n_pos = []
                out_of_map_n_pos = []
                # if we are in a terminal transition (treasure found)
                # beam back to start_state
                if self._flat_map[pos_index] > 0:
                    self.P[pos_index, :, self._index_terminal_state] = 1.0
                elif self._flat_map[pos_index] < 0:
                    self.P[pos_index, :, pos_index] = 1.0
                else:
                    # nonterminal transitions
                    for a in xrange(self.n_actions):
                        n_pos = pos + self.actions[a]
                        n_pos_index = self._get_index(n_pos)
                        if self._in_map(n_pos):
                            if self._flat_map[n_pos_index] > -100:
                                valid_n_pos.append((n_pos_index, a))
                            else:
                                out_of_map_n_pos.append((n_pos_index, a))
                        else:
                            out_of_map_n_pos.append((n_pos_index, a))
                if len(valid_n_pos) > 0:
                    # prob = 1.0 / len(valid_n_pos)
                    for n_pos_index, a in valid_n_pos:
                        self.P[pos_index, a, n_pos_index] = 1.0
                # else:
                #     self.P[pos_index, :, pos_index] = 1.0
                if len(out_of_map_n_pos) > 0:
                    for n_pos_index, a in out_of_map_n_pos:
                        self.P[pos_index, a, pos_index] = 1.0
        self.P[self._index_terminal_state, :, :] = 0
        self.P[self._index_terminal_state, :, self._index_terminal_state] = 1.0
        normalizer = self.P.sum(axis=2)[:, :, np.newaxis]
        self.P /= normalizer
        self.P[np.isnan(self.P)] = 0
        # TODO: fix this checkup of probability matrices
        # assureProbabilityMatrix(self.P)

    def reset(self):
        self.state = self._start_state
        self.terminal_state = False
        self._pre_terminal_state = False
        self._time = 0
        self.last_state = self.state
        self._position = self._get_position(self.state)
        self._last_position = self._position
        self._terminal_reward = 0

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
            return np.ravel_multi_index(position, self._scene.shape)
        else:
            if my_debug: log.debug('Error: Position out of map!')
            return -1

    def _get_position(self, index):
        if index < (self._scene.shape[0] * self._scene.shape[1]):
            return np.unravel_index(index, self._scene.shape)
        else:
            if my_debug: log.debug('Error: Index out of list!')
            return -1

    def _in_map(self, position):
        return not ((position[0] < 0) or (position[0] > self._scene.shape[0] - 1) or (position[1] < 0) or
                    (position[1] > self._scene.shape[1] - 1))

    def print_map(self):
        plt.imshow(self._scene, interpolation='none')

    def _get_reward(self, state):
        r = np.zeros(self.reward_dimension)

        if self._extended_reward:
            r[state] = 1
            return r

        # -1 for all moves
        r[1] = -1.0

        if state < self.n_states - 2:
            map_value = self._flat_map[state]
        else:
            map_value = 0.0
        # we transited to the terminal state and stay there
        if self.terminal_state:
            r[0] = 0.0
        # if we transited into a treasure state the next will be the terminal state
        elif self._pre_terminal_state:
            r[0] = map_value
        else:
            if map_value > 0:
                r[0] = map_value
            elif map_value < 0:
                # print "so nicht %i" % (state)
                r[0] = 0.0
            else:
                r[0] = 0.0

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

        if self._pre_terminal_state:
            self.terminal_state = True
            self.last_state = self.state
            self.state = self._index_terminal_state
            return self._get_reward(self.state)

        last_position = np.copy(self._position) # numpy arrays are mutable -> must be copied

        if my_debug: log.debug('Position before: ' + str(self._position) + ' moving ' + str(self.actions[action]) +
                  ' (last pos: ' + str(last_position) + ')')

        if self._in_map(self._position + self.actions[action]):
            self._position += self.actions[action]
            map_value = self._flat_map[self._get_index(self._position)]
            if my_debug: log.debug('moved by' + str(self.actions[action]) + '(last pos: ' + str(last_position) + ')')
            if map_value < 0:
                self._position = last_position
                if my_debug: log.debug('Ground touched!')
            elif map_value > 0:
                if my_debug: log.debug('Treasure found! - I got a reward of ' + str(reward))
                self._pre_terminal_state = True
                self._terminal_reward = map_value
            else:
                if my_debug: log.debug('I got a reward of ' + str(reward))
        else:
            if my_debug: log.debug('Move not allowed!')

        if my_debug: log.debug('New position: ' + str(self._position))

        self._last_position = np.copy(last_position)
        self.last_state = self.state
        self.state = self._get_index(self._position)

        # predictor_action = action
        # if random.random() > self.predictor_accuracy:
        #     predictor_action = self.__invert_action(predictor_action)

        # return np.array([reward, -self._time])
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
        super(Deepsea, self).__init__(keys=['_time', '_actions', '_scene', '_energy'])

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
            ['state', '_time', '_actions', '_scene'])

        self.actions = ('left','right','none')

        # Discount Factor
        self.gamma = gamma

        self._minPosition = -1.2  # Minimum car position
        self._maxPosition = 0.6   # Maximum car position (past goal)
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
            'right': 1, # forward thrust
            'left': -1, # backward thrust
            }

        # Determine acceleration factor
        if action < len(self.actions):
            factor = map_actions[self.actions[action]] # map action to thrust factor
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0 # Default action

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

        if self._position <= self._minPosition: #and (self._velocity < 0)
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
            'left': -1, # backward thrust
            'right': 1, # forward thrust
            'none': 0,  # coasting
            }

        # Determine acceleration factor
        if action < len(self._actions):
            factor = map_actions[self._actions[action]] # map action to thrust factor
            if (self._actions[action] == 'right') or (self._actions[action] == 'left'):
                self._nb_actions += 1
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0 # Default action

        self.car_sim(factor)

        if self.terminal_state:
            return [self._default_reward, self._time]
        else:
            return [0, self._time]


class Gridworld(MORLProblem):
    """

    """
    def __init__(self, size=10, gamma=0.9):
        self.gamma = gamma

        self.actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
        self.n_actinos = len(self.actions)
        self.n_states = size * size
        self._size = size
        self.reward_dimension = self.n_states

        self.P = None
        self.R = None

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
        self._last_state = 0
        self.terminal_state = False

    def _get_index(self, position):
        return position[1] * self.scene_x_dim + position[0]

    def _get_position(self, index):
        return index % self.scene_x_dim, index // self.scene_y_dim

    def _in_map(self, pos):
        return pos[0] >= 0 and pos[0] < self.scene_x_dim and pos[1] >= 0 and pos[1] < self.scene_y_dim

    def _get_reward(self, state):
        r = np.zeros(self.reward_dimension)
        r[state] = 1.0
        return r

    def play(self):
        pass

    @property
    def scene_x_dim(self):
        return self._size

    @property
    def scene_y_dim(self):
        return self._size
