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


class PrisonersDilemma(SaveableObject):
    '''
    The classical prisoners dilemma (PD) with a default
    payout matrix of

                        prisoner 2

                          C     D

    prisoner 1      C   (R,R) (S,T)        
                    D   (T,S) (P,P)


        with T > P > R > S and R > (T + S) / 2.
    '''

    def __init__(self, T=5.0, R=3.0, P=1.0, S=0.0):
        '''
        Initialize the Prisoner's Dilemma.

        Parameters
        ----------
        T: float
            Temptation to defect.
        R: float
            Reward for mutual cooperation.
        P: float
            Punishment for mutual defection.
        S: float
            Sucker's payoff.

        default values:
            T = 5, R = 3, P = 1, S = 0
        '''

        super(PrisonersDilemma, self).__init__(
                ['T', 'R', 'P', 'S', 'payouts', 'actions'])

        self.T = T
        self.R = R
        self.P = P
        self.S = S
        self.actions = ['cooperate', 'defect']
        self.payouts = [[(R,R), (S,T)], [(T,S), (P,P)]]

    def play(self, action1, action2):
        '''
        Play the PD.

        Parameters
        ----------
        action1: int
            Action for prisoner 1.
        action2: int
            Action for prisoner 2.

        Returns
        -------
        (payout1, payout2): float 2-tuple
            Containing the payouts for prisoner 1 and 2 respectively.
        '''
        return self.payouts[action1][action2]


class ProbabilisticPrisonersDilemma(PrisonersDilemma):
    '''
    A modification of the Prisoner's Dilemma, where the action
    of the second player is not chosen by input, but randomly
    according to some cooperation probability.
    '''

    def __init__(self, T=5.0, R=3.0, P=1.0, S=0.0, coop_p=0.5):
        '''
        Initialize the probabilistic PD.

        Parameters
        ----------
        coop_p: float [0,1]
            Set the cooperation probability.

        Other parameters, see superclass.
        '''

        super(ProbabilisticPrisonersDilemma, self).__init__(T, R, P, S)

        self.coop_p = coop_p

    def play(self, action):
        '''
        Play the probabilistic PD. 

        Parameters
        ----------
        action: int
            Action for prisoner 1.

        Returns
        -------
        (payout1, payout2): float 2-tuple
            Containing the payouts for prisoner 1 and 2 respectively.
        '''
        p2action = 1
        if random.random() <= self.coop_p:
            p2action = 0
        return self.payouts[action][p2action]
