'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

from helpers import SaveableObject

import numpy as np
import matplotlib.pyplot as plt
import random


class Deepsea(SaveableObject):
    """
    This class represents a Deepsea problem.
    All the parameters should be set up on object
    creation. Then the Deepsea problem can be used
    iteratively by calling "action".
    """

    def __init__(self, map=[], actions=["up", "down", "right", "left"], state=0):
        '''
        Initialize the Deepsea problem.

        Parameters
        ----------
        map: array, Map of the deepsea landscape. Entries represent
            rewards. Invalid states get a value of "-100" (e.g. walls, ground).
            Positive values correspond to treasures.
        actions: The name of the actions: Here the directions the
            submarine can move - left, right, up, down.
        '''

        super(Deepsea, self).__init__(
            ['energy, time, actions, map'])

        self._time = 0

        self._state = state
        self._last_state = state

        # Default Map as used in general MORL papers
        self._map = np.zeros((11,10))
        self._map[2:11,0] = -100
        self._map[3:11,1] = -100
        self._map[4:11,2] = -100
        self._map[5:11,3:6] = -100
        self._map[8:11,6:8] = -100
        self._map[10,8] = -100
        # Rewards of the default map
        self._map[1,0] = 1
        self._map[2,1] = 2
        self._map[3,2] = 3
        self._map[4,3] = 5
        self._map[4,4] = 8
        self._map[4,5] = 16
        self._map[7,6] = 24
        self._map[7,7] = 50
        self._map[9,8] = 74
        self._map[10,9] = 124

        self._flat_map = np.ravel(self._map, order='C') #flat map with Fortran-style order (column-first)

        self._position = self.get_position(state)
        self._last_position = self._position

        #self._predictor_accuracy = predictor_accuracy
        #self._payouts = payouts
        self._actions = actions

    def get_index(self, position):
        if self.in_map(position):
            #raise IndexError("Position out of bounds: {}".format(position))
            #print np.ravel_multi_index(position, self._map.shape)
            return np.ravel_multi_index(position, self._map.shape)
        else:
            print('Error: Position out of map!')
            return -1


    def get_position(self, index):
        if (index < (self._map.shape[0] * self._map.shape[1])):
            return np.unravel_index(index, self._map.shape)
        else:
            print('Error: Index out of list!')
            return -1

    def in_map(self,position):
        return not((position[0] < 0) or (position[0] > self._map.shape[0]-1) or (position[1] < 0) or (position[1] > self._map.shape[1]-1))

    def print_map(self):
        plt.imshow(self._map, interpolation='none')

 #    def __str__(self):
 #        return 'Newcomb problem with\n actions:\n%s\n\
 # predictor_accuracy:\n%03f\n\
 # payouts:\n%s' % (str(self.actions), self.predictor_accuracy, self.payouts)

    # def __invert_action(self, action):
    #     '''
    #     Invert the action, if the predictor will
    #     predict wrong.
    #     '''
    #     if action == 0:
    #         return 1
    #     return 0

    def play(self, action):
        '''
        Perform an action with the submarine
        and receive reward (or not).

        Parameters
        ----------
        action: integer, Which action will be chosen
            the agent. (0: left, 1: right, 2: up, 3: down).

        Returns
        -------
        reward: reward of the current state.
        '''

        map_actions = {
            'up': np.array([-1, 0]),
            'down': np.array([1, 0]),
            'right': np.array([0, 1]),
            'left': np.array([0, -1]),
            }

        self._time += 1

        self._last_position = self._position
        print('Position before: ' + str(self._position) + ' moving ' + self._actions[action])
        #self._position += map_actions[self._actions[action]] #try new position
        if self.in_map(self._position + map_actions[self._actions[action]]):
            self._position += map_actions[self._actions[action]]
            if self._flat_map[self.get_index(self._position)] < 0:
                self._position = self._last_position
                reward = self._flat_map[self.get_index(self._position)]
                print('Ground touched!')
            else:
                reward = self._flat_map[self.get_index(self._position)]
        else:
            print('Move not allowed!')
            reward = 0

        print(self._position)

        self._last_state = self._state
        self._state = self.get_index(self._position)

        # predictor_action = action
        # if random.random() > self.predictor_accuracy:
        #     predictor_action = self.__invert_action(predictor_action)

        return reward


class DeepseaEnergy(Deepsea):
    def __init__(self, energy = 200):
        '''
        energy: integer > 0, Amount of energy the
            the submarines battery is loaded.
        '''


    def __str__(self):
        return self.__class__.__name__

    def play(self):
        pass


