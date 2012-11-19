'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

from helpers import SaveableObject

import numpy as np
import random


class Newcomb(SaveableObject):
    """
    This class represents a Newcomb problem.
    All the parameters should be set up on object
    creation. Then the Newcomb problem can be played
    either once or iteratively.
    """

    def __init__(self, predictor_accuracy=0.99,
                 payouts=np.array([[1000000, 0], [1001000, 1000]]),
                 actions=["one box", "two box"]):
        '''
        Initialize the Newcomb problem.

        Parameters
        ----------
        predictor_accuracy: real [0,1], How good the one/two box
            predictor is predicting which one you will
            choose and therefore this is at the same
            time the probability of placement of
            the million dollar conditioned on the
            chosen boxing.
        payouts: array, size (2,2). Where the payouts
            are denoted if one/two boxing was predicted
            and one/two boxing was chosen. The first
            row denotes the payouts for chosen one boxing,
            the second for two boxing. The first column
            represents the values for predicted one
            boxing, the second for predicted two boxing.
        actions: The name of the two actions (one/two
            boxing, used for debugging output purposes
            only.
        '''

        super(Newcomb, self).__init__(
            ['predictor_accuracy, payouts, actions'])

        self.predictor_accuracy = predictor_accuracy
        self.payouts = payouts
        self.actions = actions

    def __str__(self):
        return 'Newcomb problem with\n actions:\n%s\n\
 predictor_accuracy:\n%03f\n\
 payouts:\n%s' % (str(self.actions), self.predictor_accuracy, self.payouts)

    def __invert_action(self, action):
        '''
        Invert the action, if the predictor will
        predict wrong.
        '''
        if action == 0:
            return 1
        return 0

    def play(self, action):
        '''
        Play one round of the Newcomb problem. A iterated
        Newcomb problem can be obtained by simply iteratively
        playing the same problem over and over again.

        Parameters
        ----------
        action: integer, Which action will be chosen
            the agent. (0: one box, 1: two box).

        Returns
        -------
        payout: payout of the current round.
        '''
        # decide whether the predictor is correct this
        # round
        # TODO: if the predictor is not correct does it
        #       predict a wrong answer or just a random one

        predictor_action = action
        if random.random() > self.predictor_accuracy:
            predictor_action = self.__invert_action(predictor_action)

        return self.payouts[action, predictor_action]


class RandomNewcomb(Newcomb):
    '''
    A newcomb problem which doesn't decide always the
    opposite, but instead a random boxing.
    '''

    def __invert_action(self, action):
        '''
        Play random action instead of
        the opposite of the optimal
        '''
        return random.randint(0, len(self.actions) - 1)
