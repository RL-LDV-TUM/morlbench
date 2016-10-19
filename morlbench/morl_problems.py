#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
@author: Simon Wölzmüller   <ga35voz@mytum.de>

    Copyright (C) 2016  Dominik Meyer, Johannes Feldmaier, Simon Wölzmüller

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

from helpers import SaveableObject, virtualFunction
from probability_helpers import sampleFromDiscreteDistribution

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sqrt
import logging as log
import random
import sys

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

        self.state = self._start_state
        self.terminal_state = False
        self.treasure_state = False
        self._time = 0
        self.last_state = self.state
        self._position = self._get_position(self.state)
        self._last_position = self._position

        # build state transition matrix P_{ss'} where (i, j) is the transition probability
        # from state i to j
        if self.P is None:
            self._construct_p()

        # build reward vector R(s)
        if self.R is None:
            self._construct_r()

    @staticmethod
    def name():
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
    """
    In this problem, a car tries to escape a valley by timed accelerations
    Its own force is not enough to escape only by one acceleration
    This MO version of the problem has three reward components
    """
    def __init__(self, acc_fac=0.001, cf=0.0025, gamma=0.9):
        """
        Initialize the Mountain car problem.
        reward: [time, left, right]
        every time step: time=-1
        every reversal: left=-1
        every front acceleration: right = -1
        reaching goal position: time = 100
        Parameters
        ----------
        acc_fac:    factor for acceleration function in car_sim
        cf:         cosine influence for acceleration function in car_sim
        gamma:      discount factor for this problem
        """

        super(MountainCar, self).__init__(
                ['state', '_time', 'actions', '_scene'])

        self.actions = ('left', 'none', 'right')
        self.n_actions = 3

        self.n_actions_print = self.n_actions - 1
        self.P = None
        # Discount Factor
        self.gamma = gamma

        self._nb_actions = 0  # counter for acceration actions
        self._minPosition = -1.2  # Minimum car position
        self._maxPosition = 0.6  # Maximum car position (past goal)
        self._maxVelocity = 0.07  # Maximum velocity of car
        self._goalPosition = 0.5  # Goal position - how to tell we are done
        self._accelerationFactor = acc_fac  # discount for accelerations
        self._maxGoalVelocity = 0.1
        self.n_vstates = 10.0  # for state discretization
        # continouus step for one discrete
        self.v_state_solution = (self._maxVelocity - (-self._maxVelocity))/self.n_vstates
        self.n_xstates = 28.0
        # continouus step for one discrete
        self.x_state_solution = (self._maxPosition - self._minPosition)/self.n_xstates
        self._xstates = np.arange(self._minPosition, self._maxPosition+self.x_state_solution,
                                  self.x_state_solution)
        self._vstates = np.arange(-self._maxVelocity, self._maxVelocity+self.v_state_solution,
                                  self.v_state_solution)
        self.states = []
        for x in self._xstates:
            for v in self._vstates:
                self.states.append([x, v])
        self.n_states = int((self.n_xstates+1)*(self.n_vstates+1))
        self.init_state = self.get_state([-0.5, 0.0])  # initial position ~= -0.5 ^= 32
        self.state = self.init_state      # initialize state variable
        self.last_state = self.state       # at the beginning we have no other state
        self._default_reward = 100      # reward for reaching goal position
        self.terminal_state = False     # variable for reaching terminal state
        self.n_states_print = self.n_xstates
        self._goalxState = 0.5
        self.reward_dimension = 3
        self.time_token = []
        self.cosine_factor = cf
        # self._construct_p()
        self.acceleration = 0
        self._time = 0

    def create_plottable_states(self, states):
        """
        after learning we need the 1 dim states, to plot the way through the map
        :param states: states in 2 dimensions
        :return: plottable states
        """
        plt_states = []
        for moves in states:
            plt_mvs = []
            for sts in moves:
                plt_mvs.append(self.states[sts][0])
            plt_states.append(np.array(plt_mvs))

        return np.array(plt_states)

    def reset(self):
        """
        reset the agent
        :return: nothing
        """
        self.state = self.init_state
        self.last_state = self.init_state
        self._nb_actions = 0
        self.terminal_state = False
        self._time = 0

    @staticmethod
    def distance(point1, point2):
        """
        computes a euclidean distance between point1 and point 2
        :param point1: point 1
        :param point2: point 2
        :return:
        """
        components = []
        for dim in xrange(len(point1)):
            components.append((point1[dim]-point2[dim])**2)
        return sqrt(sum(components))

    def get_state(self, position):
        """
        gets the index of a 2 dim state
        :param position: (x, v)
        :return: index
        """
        distances = [self.distance(position, self.states[i]) for i in xrange(len(self.states))]
        return np.argmin(distances)

    @staticmethod
    def name():
        """
        returns the name of the problem
        :return: name
        """
        return "Mountain Car"

    def get_velocities(self, states):
        """
        returns an array that contains the velocities [-0.07, 0.07]
        :param states: indices
        :return: real   velocities
        """
        plt_states = []
        for moves in states:
            plt_mvs = []
            for sts in moves:
                plt_mvs.append(self.states[sts][1])
            plt_states.append(np.array(plt_mvs))

        return np.array(plt_states)

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
            factor = map_actions[self.actions[action]]  # map action to thrust factor
            self.acceleration = factor
        else:
            print 'Warning: No matching action - Default action was selected!'
            factor = 0  # Default action
        # apply that factor on the car movement
        self.car_sim(factor)
        return self._get_reward(self.state)

    def _get_reward(self, state):
        """
        returns the reward for reaching state
        :param state: state we're in
        :return: reward signal of that state
        """
        reward = np.zeros(self.reward_dimension)
        # check if we reached goal
        if self.terminal_state:
            # reward!
            reward[0] = self._default_reward
        if self.acceleration == 1:
            reward[1] = -1
        if self.acceleration == -1:
            reward[2] = -1
        return reward

    def _construct_p(self):
        """
        builds a Probability function, the probability of reaching s' from state s with action a
        :return:
        """
        raise NotImplementedError("Probability matrix not implemented yet.")
        # TODO: Prepare Mountain car environment for Model based algorithms
        # self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        pass

    def car_sim(self, factor):
        """
        drive the car
        :param factor: right 1, no acc 0 and left -1
        :return:
        """

        x, v = self.states[self.state][0], self.states[self.state][1]
        # State update

        velocity_change = self._accelerationFactor * factor - self.cosine_factor * cos(3 * x)
        v += velocity_change
        # compute velocity
        if v < - self._maxVelocity:
            v = -self._maxVelocity

        elif v > self._maxVelocity:
            v = self._maxVelocity

        # add that little progress to position
        x += v
        # look if we've gone too far
        if x < self._minPosition:
            x = self._minPosition
        elif x > self._maxPosition:
            x = self._maxPosition

        # check if we're on the wrong side
        if x <= self._minPosition:  # and (self._velocity < 0)
            # inelastic wall stops the car imediately
            v = 0.0
        self.state = self.get_state([x, v])
        x, v = self.states[self.state]

        # check if we've reached the goal
        if x >= self._goalxState:
            self.terminal_state = True
            # store time token for this
            self.time_token.append(self._time)


class MountainCarTime(MountainCar):
    """
    same problem as above, just other reward structure. we get -1 every step,
    -1 for every acceleration, and +1 for reaching goal position
    """
    def __init__(self, acc_fac=0.00001, cf=0.0002):
        """
        Initialize the Multi Objective Mountain car problem.

        Parameters
        ----------
        acc_fac: factor for acceleration function in car_sim
        cf: amplitude of cosine influence on that acceleration function
        """
        self._nb_actions = 0
        super(MountainCarTime, self).__init__(acc_fac=acc_fac, cf=cf)
        self.reward_dimension = 3

    def reset(self):
        """
        prepare environment for next episode
        :return: nothing
        """
        self.state = self.init_state
        self.last_state = self.init_state
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
        """
        rewards special states
        :param state: state were in
        :return: reward vector
        """
        reward = np.zeros(self.reward_dimension)
        if self._nb_actions >= 20:
            # negative if we're accelerating to much
            reward[2] = -1

        if self.terminal_state:
            # positive reward if we are at the end
            reward[0] = self._default_reward
        else:
            # negative if we're too long in the environment
            reward[1] = -1
        return reward


class Gridworld(MORLProblem):
    """
    Original Simple-MORL-Gridworld.
    """
    def __init__(self, size=10, gamma=0.9):
        """
        constructor
        :param size: size of the grid, it will contain size*size fields
        :param gamma: discount factor for that problem
        """
        super(Gridworld, self).__init__()

        self.gamma = gamma
        # available actions:        down,           right,              up,                 left.
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
        self.state = 0
        self.last_state = 0
        self.terminal_state = False

        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    def _construct_p(self):
        """
        will construct the probability matrix
        :return: nothing
        """
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
        """
        prepares environment for next episode
        :return: nothing
        """
        self.state = 0
        self.last_state = 0
        self.terminal_state = False

    def _get_index(self, position):
        """
        returns the index of an two dimensional position vector (y,x)
        :param position: vector (y,x)
        :return: index
        """
        # return position[1] * self.scene_x_dim + position[0]
        return position[0] * self.scene_x_dim + position[1]

    def _get_position(self, index):
        """
        returns the two dimensional position (y,x)
        :param index: the index of that field we want to have
        :return: position(y,x)
        """
        return index // self.scene_x_dim, index % self.scene_x_dim

    def _in_map(self, pos):
        """
        are we still in the field?
        :param pos: (y,x)
        :return: True (in field) or False (not in field)
        """
        return 0 <= pos[0] < self.scene_x_dim and 0 <= pos[1] < self.scene_y_dim

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
        super(MORLGridworld, self).__init__()
        self.gamma = gamma

        # self.actions = (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]))
        # available actions:        right           down        left                    up
        self.actions = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))
        self.n_actions = len(self.actions)
        self.n_actions_print = self.n_actions
        self.n_states = size * size
        self.n_states_print = self.n_states
        self._size = size
        self.reward_dimension = 3
        self.last_state = self.init_state
        self.state = self.init_state
        self.P = None
        self.R = None

        # Default Map as used in general MORL papers
        self._scene = np.zeros((size, size))
        self._scene[0, size-1] = 1
        self._scene[size-1, 0] = 1
        self._scene[size-1, size-1] = 1
        self.terminal_state = False
        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    @staticmethod
    def name():
        """
        short name
        :return: simple string of name
        """
        return "MORL_Gridworld"

    def _get_reward(self, state):
        """
        rewards a specified state
        :param state: state we're in
        :return: reward vector for that state
        """
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
        """
        simulate action
        :param action: one of the four specified actions
        :return: reward for that action
        """
        actions = self.actions
        state = self.state

        position = self._get_position(state)
        n_position = position + actions[action]
        # check if we gone out of boundaries
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
        """
        constructor
        :param size: size of the grid
        :param gamma: discount factor for this problem
        """
        super(MORLGridworldTime, self).__init__()
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
        self.last_state = self.init_state
        self.state = self.init_state
        self.terminal_state = False
        if not self.P:
            self._construct_p()

        if not self.R:
            self._construct_r()

        self.reset()

    def _get_reward(self, state):
        """
        get reward from current state
        :param state: current state
        :return: reward vector
        """
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
        """
        do that action
        :param action: action to perform
        :return: reward of this action
        """
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
        """
        Constructor
        :param size: size of the grid
        :param gamma: discount rate
        """
        super(MORLGridworldStatic, self).__init__()
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
        """
        get reward from current state
        :param state: current state
        :return: reward vector
        """
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
        """
        perform defined action
        :param action: action to perform
        :return: reward vector for that action
        """
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
        self._gamma = gamma
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
        self.state = 0
        self.last_state = 0
        self.terminal_state = False
        self.reset()

        # build state transition matrix P_{ss'} where (i, j) is the transition probability
        # from state i to j
        if self.P is None:
            self._construct_p()

        # build reward vector R(s)
        if self.R is None:
            self._construct_r()

    @staticmethod
    def name():
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
        self.state = 0
        self.last_state = 0
        self.terminal_state = False
        self.reset()

        # build state transition matrix P_{ss'} where (i, j) is the transition probability
        # from state i to j
        if self.P is None:
            self._construct_p()

        # build reward vector R(s)
        if self.R is None:
            self._construct_r()

    @staticmethod
    def name():
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
    @author: Simon Wölzmüller <ga35voz@mytum.de>
    it will be rewarded with following criteria: hunger, lost food, walking distance
    hunger means
    Caution, this Problem doesn't fit on convex Hull Value iteration
    """

    def __init__(self, size=3, p=0.9, n_appear=10, gamma=0.9):
        super(MORLBuridansAss1DProblem, self).__init__(['state', '_actions', '_scene'])

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

    @staticmethod
    def name():
        return "BuridansAss1D"

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
                reward[0] = -1.0
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

    @staticmethod
    def _get_distance(state1, state2):
        first = np.array([state1[0], state1[1]])
        second = np.array([state2[0], state2[1]])
        return np.linalg.norm(second-first)

    def _get_index(self, position):
        return position[0] * self.scene_x_dim + position[1]

    def _get_position(self, index):
        return index // self.scene_y_dim, index % self.scene_x_dim

    def _in_map(self, pos):
        return 0 <= pos[0] < self.scene_x_dim and 0 <= pos[1] < self.scene_y_dim

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
    @author: Simon Wölzmüller <ga35voz@mytum.de>
    it will be rewarded with following criteria: hunger, lost food, walking distance
    hunger means that after 9 steps without eating a food pile the next action(s) will be rewarded with -1
    """

    def __init__(self, size=3, p=0.9, n_appear=10, gamma=1.0):
        """
        Constructor
        :param size: size of the cartesian coordinate field (default: 3
        :param p: probability, that food is stolen
        :param n_appear: number of steps until new food is generated after stealing
        :param gamma: discount factor for this problem
        """
        super(MORLBuridansAssProblem, self).__init__(
            ['state', '_actions', '_scene'])

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
        self.rev_food_map = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
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
        """
        after learing we need to map the states to the cartesian coordinates
        :param states: multidimensional state indices
        :return: states in 2 dim indices
        """
        pos = [[self._get_position(s) for s in states[i]] for i in xrange(len(states))]
        plt_states = [[self.position_map[p[0], p[1]] for p in pos[l]] for l in xrange(len(pos))]
        ret_states = [np.array(plt_states[i]) for i in xrange(len(plt_states))]
        ret_states = np.array(ret_states)
        return ret_states

    def _construct_p(self):
        """
        create Probability matrix
        :return: nothing
        """
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            pos, food, hunger = self._get_position(i, complete=True)
            f1, f2 = self.rev_food_map[food][0], self.rev_food_map[food][1]
            y, x = self.get_cartesian_coordinates_from_pos_state(pos)
            for a in xrange(self.n_actions):
                ax, ay = self.actions[a]
                nx, ny = x+ax, y+ay
                if a != 0:
                    new_hunger = hunger + 1 if (hunger >= self.max_hunger) else self.max_hunger - 1
                else:
                    new_hunger = hunger
                if not self._in_map((ny, nx)):
                    new_state = self.state_map[pos, food, new_hunger]
                    self.P[i, a, new_state] = 1.0
                    continue
                n_pos_s = self.position_map[ny, nx]

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
                        self.P[i, a, new_s] = 1.0
                if (y, x) == self.food2:
                    if a == 0:
                        new_f_st = self.food_map[f1, 0]
                        new_s = self.state_map[pos, new_f_st, 0]
                        self.P[i, a, new_s] = 1.0
                else:
                    new_state = self.state_map[n_pos_s, food, new_hunger]
                    self.P[i, a, new_state] = 1.0

    def get_cartesian_coordinates_from_pos_state(self, state):
        """
        multidimensional --> two dimensional state positions
        :param state: 3D coordinates (pos, hunger, food)
        :return: position (y, x)
        """
        for o in self.position_map.keys():
            if self.position_map[o] == state:
                cart_pos = o
        return cart_pos[0], cart_pos[1]

    def reset(self):
        """
        prepare environment for next episode
        :return:
        """
        self.state = self.init_state
        self.last_state = self.init_state
        self.terminal_state = False
        self.count = 0
        self.hunger = 0
        self._scene[self._size-1, self._size-1] = 1
        self._scene[0, 0] = 1

    @staticmethod
    def name():
        """
        return simple name
        :return: name
        """
        return "BuridansAss"

    def _get_reward(self, state):
        """
        create reward vector on current state
        :param state: current state
        :return: reward vector
        """
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
        """
        perform this action
        :param action: this action
        :return: reward for this action
        """
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

    @staticmethod
    def _get_distance(state1, state2):
        """
        compute distance between two points
        :param state1: first point
        :param state2: second point
        :return: double distance
        """
        first = np.array([state1[0], state1[1]])
        second = np.array([state2[0], state2[1]])
        return np.linalg.norm(second-first)

    def _get_index(self, position):
        """
        get the index of a cartesian position
        :param position: position (y, x)
        :return: index
        """
        x = sys._getframe(1)
        if x.f_code.co_name == '_policy_plot2':
            return position[0] * self.scene_x_dim + position[1]
        pos = self.position_map[position[1], position[0]]
        f = self.food_map[self.food[0], self.food[1]]
        h = self.hunger
        return self.state_map[pos, f, h]

    def _get_position(self, index, complete=False):
        """
        get multidimensional position, complete: pos, hunger, food
        not complete: pos (y,x)
        :param index: index for which we need the position
        :param complete: do we need complete(True) state position or not(False)
        :return: complete or not complete position
        """
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
        """
        are we still in boundaries
        :param pos: coordinates
        :return: in map (true) or not in map (False)
        """
        return 0 <= pos[0] < self.scene_y_dim and 0 <= pos[1] < self.scene_x_dim

    @property
    def scene_x_dim(self):
        """
        x dimensional states count
        :return: count
        """
        return self._size

    @property
    def scene_y_dim(self):
        """
        y dimensional states count
        :return: count
        """
        return self._size

    def print_map(self, pos=None):
        """
        show a plot of that scene
        :param pos: if you want to highlight one field, give it.
        :return: nothing
        """
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()


class MOPuddleworldProblem(MORLProblem):
    """
    This problem contains a quadratic map (please use size more than 15, to get a useful puddle)
    the puddle is an obstacle that the agent has to drive around. The aim is to reach the goal state at the top right
    @author: Simon Wölzmüller <ga35voz@mytum.de>
    """
    def __init__(self, size=20, gamma=0.9):
        """
        Constructor
        :param size: size of grid
        :param gamma: discount factor of this problem
        """
        super(MOPuddleworldProblem, self).__init__(['state', '_actions', '_scene'])
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
            self._scene[4:6, 3:10] = -4.0
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

        self.P = None
        self._construct_p()
        self.R = None
        self._construct_r()

    def _set_new_init(self):
        """
        This problem chooses random state every episode
        :return: nothin
        """
        self.init_state = random.choice(self.init)

    def plot_map(self):
        """
        plot a map of cartesian states
        :return: a array of the flat map
        """
        # plot
        fig, ax = plt.subplots()
        temp = self._scene
        ax.imshow(temp, interpolation='nearest')
        # step = 1.
        mino = 0.
        rows = temp.shape[0]
        columns = temp.shape[1]
        row_arr = np.arange(mino, rows)
        col_arr = np.arange(mino, columns)
        x, y = np.meshgrid(row_arr, col_arr)
        for col_val, row_val in zip(x.flatten(), y.flatten()):
            c = int(temp[row_val, col_val])
            ax.text(col_val, row_val, c, va='center', ha='center')
        flat_map = np.argwhere(self._scene >= 0)  # get all indices greater than zero
        # get all elements greater than zero and stack them to the corresponding index
        flat_map = np.column_stack((flat_map,  self._scene[self._scene >= 0]))
        plt.show()
        return flat_map

    def _construct_p(self):
        """
        builds the Probabilty matrix
        :return:
        """
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in xrange(self.n_states):
            for a in xrange(self.n_actions):
                sy, sx = self._get_position(i)
                ay, ax = self.actions[a][0], self.actions[a][1]
                nsy, nsx = sy+ay, sx +ax
                if not self._in_map((nsy, nsx)):
                        self.P[i, a, i] = 1.0
                else:
                    j = self._get_index((nsy, nsx))
                    self.P[i, a, j] = 1.0

    def reset(self):
        """
        prepare environment for new episode
        :return:
        """
        self._set_new_init()
        self.state = self.init_state
        self.last_state = self.init_state
        self.terminal_state = False

    def _get_reward(self, state):
        """
        get the reward vector for this state
        :param state: this state
        :return: reward vector
        """
        position = self._get_position(state)
        reward = np.zeros(self.reward_dimension)
        if self._in_map(position) and self._scene[position] < 0:
            reward[1] = self._scene[position]*10
        if self._scene[position] > 0:
            reward[0] = 10
        else:
            reward[0] = -1

        return reward

    @staticmethod
    def name():
        """
        returns a simple name string
        :return: name
        """
        return "Puddleworld"

    def play(self, action):
        """
        perform an action
        :param action: action to perform
        :return: reward for this action
        """
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
        """
        converts y,x into index
        :param position: y, x
        :return: index
        """
        y, x = position[0], position[1]
        return self.state_map[y, x]

    def _get_position(self, index):
        """
        converts index in y x coordinates
        :param index: input index
        :return: coordinates
        """
        return index // self.scene_x_dim, index % self.scene_x_dim

    def _in_map(self, pos):
        """
        are we still in boundaries
        :param pos: coordinates
        :return: in map (true) or not in map (False)
        """
        return 0 <= pos[0] < self.scene_y_dim and 0 <= pos[1] < self.scene_x_dim

    @property
    def scene_x_dim(self):
        """
        x dimensional states count
        :return: count
        """
        return self._size

    @property
    def scene_y_dim(self):
        """
        y dimensional states count
        :return: count
        """
        return self._size

    def print_map(self, pos=None):
        """
        plots a map of states and grid
        :param pos:
        :return:
        """
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()


class MORLResourceGatheringProblem(MORLProblem):
    """
    In this problem the agent has to find the resources and bring them back to the homebase.
    the enemies steal resources with a probability of 0.1

    """
    def __init__(self, size=5, gamma=0.9, p=0.1):
        """
        Constructor
        :param size: size of the grid
        :param gamma: discount rate of the problem
        :param p: probability of enemy attack
        """
        super(MORLResourceGatheringProblem, self).__init__(['state', '_actions', '_scene'])
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
        self.n_bag_place_states = 4
        self.n_states = size * size * self.n_bag_place_states
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
        self.stolen = False
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
        self.brought_home = 0

    def get_bag_index(self, bag):
        """
        get the index of bag state: 0:00, 1:01, 2:10, 3:11
        :param bag: bagstate
        :return: index
        """
        for i in self._bag_mapping.keys():
            if self._bag_mapping[i] == bag:
                return i

    def reset(self):
        """
        prepare environment for next episode
        :return:
        """
        self.init = 88
        self.state = self.init
        self.last_state = self.init
        self.terminal_state = False
        self._scene[0, 2] = 1
        self._scene[1, 4] = 1

    def create_plottable_states(self, states):
        """
        after learning, convert states into 2 dim. map states
        :param states: the multidimensional states
        :return: 2 dim states
        """
        pos = [[self._get_position(s) for s in states[i]] for i in xrange(len(states))]
        plt_states = np.array([np.array([self.position_map[p[0], p[1]] for p in pos[l]]) for l in xrange(len(pos))])
        ret_states = [np.array(plt_states[i]) for i in xrange(len(plt_states))]
        ret_states = np.array(ret_states)
        self.n_states = self.n_states_print
        return np.array(ret_states)

    def _get_reward(self, state):
        """
        get the reward of this state
        :param state: this state
        :return: reward vector
        """
        pos = self._get_position(state)
        position = pos[0], pos[1]
        reward = np.zeros(self.reward_dimension)

        # if we're turning back home
        if [position[0], position[1]] == self.init_position:
            if self.stolen:
                reward[0] = -10
                self.stolen = False
                # we need to get back to the homebase
                return reward
            # we get reward for the bag (0,1,0), (0,1,1), (0,0,1), or (0,0,0)
            if self.brought_home:
                reward[1:] = self._bag_mapping[self.brought_home]
                self.brought_home = 0
        if (reward > 0).any():
            self.terminal_state = True
        return reward

    def _construct_p(self):
        """
        build a probability function
        :return: nothing
        """
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

    @staticmethod
    def name():
        """
        return simple name string
        :return: string name
        """
        return "Resource Gathering"

    def _construct_r(self):
        """
        construct reward function
        :return: nothing
        """
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
        """
        perform one action
        :param action: index of the action
        :return: reward vector of that action
        """
        actions = self.actions
        state = self.state
        position = self._get_position(state)[:2]
        bag = self._get_position(state)[2:]
        n_position = position + actions[action]
        if not self._in_map(n_position):
            self.state = state
            self.last_state = state
            reward = self._get_reward(self.state)
            return reward
        if self._scene[n_position[0], n_position[1]] < 0 and random.random() < self.losing_probability:
            # our resources is stolen
            self._bag[:] = [0, ] * len(self._bag)
            self.stolen = True
            print 'stolen'
            # we need to get back to the homebase
            position = self.init_position
            self.state = self._get_index((position[0], position[1], self._bag[0], self._bag[1]))
            reward = self._get_reward(self.state)
            return reward
        # we are 1. in the map and 2. we found resource:
        if self._scene[n_position[0], n_position[1]] > 0:  # put the resource in our bag
            # check which resource it is
            if (n_position == self.resource_positions[0]).all():
                self._bag[0] = 1
            if (n_position == self.resource_positions[1]).all():
                self._bag[1] = 1
            # delete it from the field
            self._scene[n_position] = 0

        if (n_position == self.init_position).all():
            self.brought_home = self.get_bag_index(self._bag)
            self._bag[0] = 0
            self._bag[1] = 0

        self.state = self._get_index((n_position[0], n_position[1], self._bag[0], self._bag[1]))
        self.last_state = state

        reward = self._get_reward(self.state)
        return reward

    def _get_index(self, position):
        """
        get the index for a position 3 dim --> 2 dim
        :param position: y, x, bag1, bag2
        :return: index
        """
        x = sys._getframe(1)
        if x.f_code.co_name == '_policy_plot2':
            return position[0] * self.scene_x_dim + position[1]
        bagind = self.get_bag_index([position[2], position[3]])
        return (position[0] * self.scene_x_dim + position[1])*len(self._bag_mapping) + bagind

    def _get_position(self, index):
        """
        get the y, x, bag1, bag2 from the index
        :param index: input
        :return: y, x, bag1, bag2
        """
        bag_state = index % len(self._bag_mapping)
        bag = self._bag_mapping[bag_state]
        index /= len(self._bag_mapping)
        return index // self.scene_x_dim, index % self.scene_y_dim, bag[0], bag[1]

    def _in_map(self, pos):
        """
        if we're outta bounds, return False
        :param pos: position
        :return: True for in map, False for outta map
        """
        return 0 <= pos[0] < self.scene_y_dim and 0 <= pos[1] < self.scene_x_dim

    @property
    def scene_x_dim(self):
        """
        count of x states
        :return: count
        """
        return self._size

    @property
    def scene_y_dim(self):
        """
        count of y states
        :return: count
        """
        return self._size

    def print_map(self, pos=None):
        """
        print a state map
        :param pos: highlighted field
        :return: nothing
        """
        tmp = self._scene
        if pos:
            tmp[tuple(pos)] = tmp.max() * 2.0
        plt.imshow(self._scene, interpolation='nearest')
        plt.show()
