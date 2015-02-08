'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

from helpers import virtualFunction, SaveableObject

import numpy as np
import random
import logging as log

#log.basicConfig(level=log.DEBUG)


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

    def interact(self):
        '''
        Interact once, the internal state
        has to be maintained by the child class
        itself.
        '''

        virtualFunction()

    def interact_multiple(self, n=1):
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

class EUNewcombAgent(NewcombAgent):
    '''
    A Newcomb Agent, that decides according to the
    calcuated expected utility. We assume, that the
    agent has access to the prediction accuracy of
    the superhuman intelligence.
    '''

    def __init__(self, problem):
        super(EUNewcombAgent, self).__init__(problem)

    def _decide(self, t):
        n_actions = len(self.newcomb_problem.actions)
        accuracy = self.newcomb_problem.predictor_accuracy
        payouts = self.newcomb_problem.payouts
        utility = np.zeros(n_actions)
        for a in xrange(n_actions):
            # TODO; fix for more than 2 actions
            utility[a] += accuracy * payouts[a][0] + (1.0 - accuracy) * \
                payouts[a][1]
        action = np.argmax(utility)
        return action

    def get_learned_action(self):
        return self._decide(1)


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
        self.last_action = 0
        self.last_payout = 0

    def interact_multiple(self, n=1):
        '''
        Interact n times with the problem and return
        the array of payouts.
        '''
        log.info('Playing %i interactions ... ' % (n))
        payouts = []
        self.last_payout = 0
        self.last_action = 0
        for t in xrange(n):
            action, payout = self.interact(t)
            payouts.append(payout)
            log.debug(' step %05i: action: %i, payout: %i' % \
                      (t, action, payout))
        return np.array(payouts)

    def interact(self, t):
        '''
        Interact only once with the given Newcomb problem.
        Maintain last payouts and actions internally.
        '''
        action = self._decide(t)
        payout = self.newcomb_problem.play(action)
        self._learn_sarsa(t, self.last_action,
                              self.last_payout, action, payout)
        self.last_action = action
        self.last_payout = payout
        return action, payout

    def _learn_sarsa(self, t, last_action,
                     last_payout, action, payout):
        self.Q[last_action] += self.alpha * \
            (payout + self.gamma * self.Q[action] - self.Q[last_action])
        log.debug(' Q: %s' % (str(self.Q)))

    def _decide(self, t):
        if random.random() < self.epsilon:
            action = self.Q.argmax()
            log.debug('  took greedy action %i' % (action))
            return action
        action = random.randint(0, len(self.newcomb_problem.actions) - 1)
        log.debug('   took random action %i' % (action))
        return action

    def get_learned_action(self):
        return self.Q.argmax()
    

class UCB1NewcombAgent(NewcombAgent):
    '''
    A Newcomb agent, that uses UCB1 to decide which
    boxing to do next, as described in 
    http://www.cs.mcgill.ca/~vkules/bandits.pdf
    '''

    def __init__(self, problem):
        '''
        Initialize the UCB1 Newcomb
        Agent with the problem description.

        Parameters
        ----------
        problem: A Newcomb problem
        '''
        super(UCB1NewcombAgent, self).__init__(problem)

        # the mu function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        # We will maintain the empirical means in the
        # vector mu and the number of plays for each 
        # arms in the vector n
        self.mu = np.zeros(len(self.newcomb_problem.actions))
        self.n = np.zeros(len(self.newcomb_problem.actions))
        self.total_interactions = 0

    def interact_multiple(self, n=1):
        '''
        Interact n times with the problem and return
        the array of payouts.
        '''
        log.info('Playing %i interactions ... ' % (n))
        payouts = []
        self.last_payout = 0
        self.last_action = 0
        for t in xrange(n):
            action, payout = self.interact(t)
            payouts.append(payout)
            log.debug(' step %05i: action: %i, payout: %i' % \
                      (t, action, payout))
        return np.array(payouts)

    def interact(self, t):
        '''
        Interact only once with the given Newcomb problem.
        Maintain last payouts and actions internally.
        '''
        action = self._decide(t)
        payout = self.newcomb_problem.play(action)
        self.total_interactions += 1
        self.n[action] += 1
        self.mu[action] += 1.0 / (self.n[action] + 1) * (payout - self.mu[action])
        return action, payout

    def _decide(self, t):
        action = np.argmax(self.mu + np.sqrt((2 * 
                        np.log(self.total_interactions)) / self.n))
        return action

    def get_learned_action(self):
        return self._decide(1)


class PrisonerAgent(SaveableObject):
    '''
    A agent that should interface with a Prisoner's dilemma problem.
    '''

    def __init__(self, pd):
        '''
        Initialize the Agent with the Prisoner's
        dilemma, it will be faced with.

        Parameters
        ----------
        newcomb_problem: The already initialized and
            correctly parametrized problem.
        '''

        super(PrisonerAgent, self).__init__([])

        self.pd = pd

    def __str__(self):
        return self.__class__.__name__

    def interact(self):
        '''
        Interact once, the internal state
        has to be maintained by the child class
        itself.
        '''

        virtualFunction()

    def interact_multiple(self, n=1):
        '''
        Interact n times with the problem and return
        the array of payouts.
        '''
        virtualFunction()
       
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


class ProbabilisticPrisonerAgent(PrisonerAgent):
    '''
    A PD agent, that is able to play the probabilistic
    version of the PD.
    '''

    def interact_multiple(self, n=1):
        log.info('Playing %i interactions ... ' % (n))
        payouts = []
        for t in xrange(n):
            a = self._decide(t)
            p = self.pd.play(a)
            payouts.append(p)
        return np.array(payouts)


class DefectProbabilisticPrisonerAgent(ProbabilisticPrisonerAgent):
    '''
    A PD agent that always defects.
    '''

    def _decide(self, t):
        return 1


class CooperateProbabilisticPrisonerAgent(ProbabilisticPrisonerAgent):
    '''
    A PD agent that always cooperates.
    '''

    def _decide(self, t):
        return 0


