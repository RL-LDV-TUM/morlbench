'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

from helpers import virtualFunction, SaveableObject

import numpy as np
import logging as log

log.basicConfig(level=log.DEBUG)


class NewcombAgent(SaveableObject):
    '''
    A agent that should interface with a Newcomb problem.
    '''

    def __init__(self, newcomb_problem):
        '''
        Initialize the Agent with the Newcomb
        problem, it will be faced with.

        Parameters
        ----------
        newcomb_problem: The already initialized and
            correctly parametrized problem.
        '''

        super(NewcombAgent, self).__init__([])

        self.newcomb_problem = newcomb_problem

    def __str__(self):
        return self.__class__.__name__

    def interact(self, n=1):
        '''
        Interact n times with the problem and return
        the array of payouts.
        '''
        log.info('Playing %i interactions ... ' % (n))
        payouts = []
        for t in xrange(n):
            a = self._decide(t)
            p = self.newcomb_problem.play(a)
            payouts.append(p)
        return np.array(payouts)

    def _decide(self, t):
        '''
        Decide which action to take in interaction
        cycle t.

        Parameters
        ----------
        t: Interaction cycle we are currently in

        Returns
        -------
        action: The action to do next
        '''
        virtualFunction()


class OneBoxNewcombAgent(NewcombAgent):
    '''
    A Newcomb Agent, that always chooses one boxing.
    '''

    def __init__(self, problem):
        super(OneBoxNewcombAgent, self).__init__(problem)

    def _decide(self, t):
        return 0


class TwoBoxNewcombAgent(NewcombAgent):
    '''
    A Newcomb Agent, that always chooses two boxing.
    '''

    def __init__(self, problem):
        super(TwoBoxNewcombAgent, self).__init__(problem)

    def _decide(self, t):
        return 1
