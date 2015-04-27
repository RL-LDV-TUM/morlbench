'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

from helpers import virtualFunction, SaveableObject

import numpy as np
import random
import logging as log

# log.basicConfig(level=log.DEBUG)


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

    def decide(self, t):
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

    def learn(self, t, action, payout):
        '''
        Learn from the last interaction, if we have
        a dynamically learning agent.

        Parameters
        ----------
        t: int
            Interaction cycle.
        action: int
            Last action
        payout: float
            Last recevied payout.
        '''
        virtualFunction()


class OneBoxNewcombAgent(NewcombAgent):
    '''
    A Newcomb Agent, that always chooses one boxing.
    '''

    def __init__(self, problem):
        super(OneBoxNewcombAgent, self).__init__(problem)

    def decide(self, t):
        return 0


class TwoBoxNewcombAgent(NewcombAgent):
    '''
    A Newcomb Agent, that always chooses two boxing.
    '''

    def __init__(self, problem):
        super(TwoBoxNewcombAgent, self).__init__(problem)

    def decide(self, t):
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

    def decide(self, t):
        n_actions = len(self.newcomb_problem.actions)
        accuracy = self.newcomb_problem.predictor_accuracy
        payouts = self.newcomb_problem.payouts
        utility = np.zeros(n_actions)
#         for a in xrange(n_actions):
#             # TODO; fix for more than 2 actions
#             utility[a] += accuracy * payouts[a][0] + (1.0 - accuracy) * \
#                 payouts[a][1]
        utility[0] = accuracy * payouts[0][0] + (1.0 - accuracy) * payouts[0][1]
        utility[1] = (1.0 - accuracy) * payouts[1][0] + accuracy * payouts[1][1]
        action = np.argmax(utility)
        return action

    def get_learned_action(self):
        return self._decide(1)


class SARSANewcombAgent(NewcombAgent):
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
        super(SARSANewcombAgent, self).__init__(problem)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self.Q = np.zeros(len(self.newcomb_problem.actions))
        self.last_action = 0
        self.last_payout = 0

#     def interact_multiple(self, n=1):
#         '''
#         Interact n times with the problem and return
#         the array of payouts.
#         '''
#         log.info('Playing %i interactions ... ' % (n))
#         payouts = []
#         self.last_payout = 0
#         self.last_action = 0
#         for t in xrange(n):
#             action, payout = self.interact(t)
#             payouts.append(payout)
#             log.debug(' step %05i: action: %i, payout: %i' %
#                       (t, action, payout))
#         return np.array(payouts)
# 
#     def interact(self, t):
#         '''
#         Interact only once with the given Newcomb problem.
#         Maintain last payouts and actions internally.
#         '''
#         action = self._decide(t)
#         payout = self.newcomb_problem.play(action)

    def learn(self, action, payout):
        '''
        Learn on the last interaction specified by the
        action and the payout received.
        '''
        self._learn(0, self.last_action,
                    self.last_payout, action, payout)
        self.last_action = action
        self.last_payout = payout

    def _learn(self, t, last_action, last_payout, action, payout):
        self.Q[last_action] += self.alpha * \
            (payout + self.gamma * self.Q[action] - self.Q[last_action])
        log.debug(' Q: %s' % (str(self.Q)))

    def decide(self, t):
        if random.random() < self.epsilon:
            action = self.Q.argmax()
            log.debug('  took greedy action %i' % (action))
            return action
        action = random.randint(0, len(self.newcomb_problem.actions) - 1)
        log.debug('   took random action %i' % (action))
        return action

    def get_learned_action(self):
        return self.Q.argmax()


class AVGQNewcombAgent(SARSANewcombAgent):
    '''
    A Newcomb agent, that does not use the SARSA, but instead
    the incremental average Q update rule.
    '''

    def __init__(self, *args, **kwargs):
        super(AVGQNewcombAgent, self).__init__(*args, **kwargs)

        self.n_times = np.zeros_like(self.Q)

    def _learn(self, t, last_action, last_payout, action, payout):
        self.Q[action] += 1.0 / (self.n_times[action] + 1) * (payout - self.Q[action])
        self.n_times[action] += 1


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

    def learn(self, t, action, payout):
        self.total_interactions += 1
        self.n[action] += 1
        self.mu[action] += 1.0 / (self.n[action] + 1) * (payout -
                                                         self.mu[action])
        self.last_action = action
        self.last_payout = payout

    def decide(self, t):
        action = np.argmax(self.mu +
                           np.sqrt((2 * np.log(self.total_interactions)) /
                                   self.n))
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

    def decide(self, t):
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

    def learn(self, t, action, payout):
        '''
        Learn from the last interaction, if we have
        a dynamically learning agent.

        Parameters
        ----------
        t: int
            Interaction cycle.
        action: int
            Last action
        payout: float
            Last recevied payout.
        '''
        virtualFunction()


class DefectPrisonerAgent(PrisonerAgent):
    '''
    A PD agent, that always defects.
    '''

    def decide(self, t):
        return 1

    def learn(self, t, action, payout):
        pass


class CooperatePrisonerAgent(PrisonerAgent):
    '''
    A PD agent that always cooperates.
    '''

    def decide(self, t):
        return 0

    def learn(self, t, action, payout):
        pass


class ProbabilisticPrisonerAgent(PrisonerAgent):
    '''
    A PD agent, that is able to play the probabilistic
    version of the PD.
    '''

    pass


class DefectProbabilisticPrisonerAgent(ProbabilisticPrisonerAgent,
                                       DefectPrisonerAgent):
    '''
    A PD agent that always defects.
    '''

    pass


class CooperateProbabilisticPrisonerAgent(ProbabilisticPrisonerAgent,
                                          CooperatePrisonerAgent):
    '''
    A PD agent that always cooperates.
    '''

    pass


class SARSAPrisonerAgent(ProbabilisticPrisonerAgent):
    '''
    A PD agent, that decides according to a SARSA
    RL learning stragey.
    '''

    def __init__(self, pd, alpha=0.3, gamma=1.0, epsilon=1.0):
        '''
        Initialize the Reinforcement Learning PD
        Agent with the probleme description and alpha,
        the learning rate.

        Parameters
        ----------
        pd: A PD problem
        alpha: real, the learning rate in each
            SARSA update step
        gamma: real, [0, 1) RL discount factor
        epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        '''
        super(SARSAPrisonerAgent, self).__init__(pd)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self.Q = np.zeros(len(self.pd.actions))
        self.n_times = np.zeros_like(self.Q)
        self.last_action = 0
        self.last_payout = 0

    def decide(self, t):
        '''
        Alternative interface to interact with multiple
        PD agents. Use in conjunction with `learn'
        '''
        return self._decide(t)

    def learn(self, t, action, payout):
        '''
        Learn from interaction.

        Parameters
        ----------
        action: int
            The action that lead to the payout.
        payout: float
            Payout from the problem for this agent.
        '''
        self.Q[self.last_action] += self.alpha * \
            (payout + self.gamma * self.Q[action] - self.Q[self.last_action])
        log.debug(' Q: %s' % (str(self.Q)))
#         print ' Q: %s' % (str(self.Q))
        self.last_action = action
        self.last_payout = payout

    def _decide(self, t):
        if random.random() < self.epsilon:
            action = self.Q.argmax()
            log.debug('  took greedy action %i' % (action))
            return action
        action = random.randint(0, len(self.pd.actions) - 1)
        log.debug('   took random action %i' % (action))
        return action

    def get_learned_action(self):
        return self.Q.argmax()


class AVGQPrisonerAgent(SARSAPrisonerAgent):
    '''
    A PD agent, that decides according to a AVGQ
    RL learning stragey.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Initialize the AVGQ PD
        '''
        super(AVGQPrisonerAgent, self).__init__(*args, **kwargs)

        self.n_times = np.zeros_like(self.Q)

    def learn(self, action, payout):
        '''
        Learn from interaction.

        Parameters
        ----------
        action: int
            The action that lead to the payout.
        payout: float
            Payout from the problem for this agent.
        '''
        self.Q[action] += 1.0 / (self.n_times[action] + 1) * (payout -
                                                              self.Q[action])
        self.n_times[action] += 1
        log.debug(' Q: %s' % (str(self.Q)))
#         print ' Q: %s' % (str(self.Q))
        self.last_action = action
        self.last_payout = payout


class EUPrisonerAgent(ProbabilisticPrisonerAgent):
    '''
    A PD agent, that decides according to expected utility
    RL learning stragey.
    '''

    def __init__(self, pd):
        '''
        Initialize the EU PD
        Agent with the probleme description and alpha,
        the learning rate.

        Parameters
        ----------
        pd: A PD problem
        '''
        super(EUPrisonerAgent, self).__init__(pd)

    def decide(self, t, total_payout=False):
        '''
        Alternative interface to interact with multiple
        PD agents. Use in conjunction with `learn'
        '''
        return self._decide(t, total_payout)

    def learn(self, action, payout):
        '''
        Learn from interaction.

        EU does not learn from interaction.
        '''
        pass

    def _decide(self, t, total_payout):
        n_actions = len(self.pd.actions)
        accuracy = self.pd.coop_p
        payouts = self.pd.payouts
        utility = np.zeros(n_actions)
        # TODO: make this work for general problems
        if total_payout:
            utility[0] = accuracy * (payouts[0][0][0] + payouts[0][0][1]) + \
                (1.0 - accuracy) * (payouts[0][1][0] + payouts[0][1][1])
            utility[1] = (1.0 - accuracy) * (payouts[1][1][0] + payouts[1][1][1]) + \
                accuracy * (payouts[1][0][0] + payouts[1][0][1])
        else:
            utility[0] = accuracy * payouts[0][0][0] + (1.0 - accuracy) * payouts[0][1][0]
            utility[1] = (1.0 - accuracy) * payouts[1][1][0] + accuracy * payouts[1][0][0]
        action = np.argmax(utility)
        return action

    def get_learned_action(self, total_payout=False):
        return self._decide(1, total_payout)

