'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

from helpers import virtualFunction, SaveableObject

import numpy as np
import random
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


class RLNewcombAgent(NewcombAgent):
    '''
    A Newcomb agent, that uses RL to decide which
    boxing to do next.
    '''

    def __init__(self, problem, alpha=0.3, gamma=1.0, epsilon=1.0):
        '''
        Initialize the Reinforcement Learning Newcomb
        Agent with the probleme description and alpha,
        the learning rate.

        Parameters
        ----------
        problem: A Newcomb problem
        alpha: real, the learning rate in each
            SARSA update step
        gamma: real, [0, 1) RL discount factor
        epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        '''
        super(RLNewcombAgent, self).__init__(problem)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self.Q = np.zeros(len(self.newcomb_problem.actions))

    def interact(self, n=1):
        '''
        Interact n times with the problem and return
        the array of payouts.
        '''
        log.info('Playing %i interactions ... ' % (n))
        payouts = []
        last_payout = 0
        last_action = 0
        for t in xrange(n):
            action = self._decide(t)
            payout = self.newcomb_problem.play(action)
            self._learn_sarsa(t, last_action,
                              last_payout, action, payout)
            payouts.append(payout)
            last_action = action
            last_payout = payout
            log.debug(' step %05i: action: %i, payout: %i' % \
                      (t, action, payout))
        return np.array(payouts)

    def _learn_sarsa(self, t, last_action,
                     last_payout, action, payout):
        self.Q[last_action] += self.alpha * \
            (payout + self.gamma * self.Q[action] - self.Q[last_action])
        log.debug(' Q: %s' % (str(self.Q)))

    def _decide(self, t):
        if random.random() < self.epsilon:
            return self.Q.argmax()
        return random.randint(0, len(self.newcomb_problem.actions) - 1)
