#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
"""

from helpers import virtualFunction, SaveableObject, HyperVolumeCalculator, remove_duplicates

import numpy as np
import random
import logging as log
try:
    # import neurolab only if it exists in case it is not used and not installed
    # such that the other agents still work
    import neurolab as nl
except ImportError, e:
    log.warn("Neurolab not installed: %s" % (str(e)))

# log.basicConfig(level=if my_debug: log.debug)
my_debug = log.getLogger().getEffectiveLevel() == log.DEBUG


class MorlAgent(SaveableObject):
    """
    A agent that should interface with a MORL problem.
    """

    def __init__(self, morl_problem, **kwargs):
        """
        Initialize the Agent with the MORL problem
        problem, it will be faced with.

        Parameters
        ----------
        morl_problem: The already initialized and
            correctly parametrized problem.
        """

        super(MorlAgent, self).__init__([])

        self._morl_problem = morl_problem
        # Discount Factor is part of the problem, but is used in most algorithms
        self._gamma = morl_problem.gamma

    def __str__(self):
        return self.__class__.__name__

    def name(self):
        return str(self.__class__.__name__)

    def learn(self, t, last_state, action, reward, state):
        """
        Learn on the last interaction specified by the
        action and the reward received.

        :param t: Interaction cycle we are currently in
        :param last_state: The last state where we transited from
        :param action: last interaction action
        :param reward: received reward vector
        :param state: next state transited to
        :return:
        """
        # virtualFunction()

    def decide(self, t, state):
        """
        Decide which action to take in interaction
        cycle t.

        Parameters
        ----------
        :param t: Interaction cycle we are currently in
        :param state: state we are in

        Returns
        -------
        action: The action to do next
        """
        virtualFunction()

    def learn(self, t, last_state, action, reward, state):
        """
        Learn from the last interaction, if we have
        a dynamically learning agent.

        Parameters
        ----------
        :param t: int Interaction cycle.
        :param last_state: Last state where we came from
        :param action: last interaction action
        :param reward: received reward vector
        :param state: next state transited to
        """
        virtualFunction()


class TDMorlAgent(MorlAgent):
    """
    A MORL agent, that uses TD for Policy Evaluation.
    """

    def __init__(self, problem, scalarization_weights, policy, alpha=0.3, **kwargs):
        """
        Initialize the TD Policy Evaluation learner for MORL.
        Scalarization weights have to be given.

        :param problem: MORL problem.
        :param scalarization_weights: Reward scalarization weights.
        :param policy: A static policy, that will be evaluated.
        :param alpha: Learning rate.
        """
        super(TDMorlAgent, self).__init__(problem, **kwargs)

        self._policy = policy
        self._scalarization_weights = scalarization_weights
        self._alpha = alpha

        self._V = np.zeros(self._morl_problem.n_states)

    def learn(self, t, last_state, action, reward, state):
        scalar_reward = np.dot(self._scalarization_weights.T, reward)
        self._V[last_state] += self._alpha * (scalar_reward + self._gamma * self._V[state] - self._V[last_state])

        if my_debug:
            log.debug(' V: %s' % (str(self._V[0:110].reshape((11, 10)))))

    def decide(self, t, state):
        return self._policy.decide(state)

    def reset(self):
        """
        resets the current agent! Be careful and save value function first!
        """
        self._V = np.zeros(self._morl_problem.n_states)


class SARSAMorlAgent(MorlAgent):
    """
    A MORL agent, that uses RL.
    """

    def __init__(self, problem, scalarization_weights, alpha=0.3, epsilon=1.0, **kwargs):
        """
        Initialize the Reinforcement Learning MORL
        Agent with the problem description and alpha,
        the learning rate.

        Parameters
        ----------
        :param problem: A MORL problem
        :param scalarization_weights: a weight vector to scalarize the morl reward.
        :param alpha: real, the learning rate in each
            SARSA update step
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(SARSAMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha

        self._epsilon = epsilon

        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        # hidden variables for conserving agent state
        self._Q_save = []
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)

    def learn(self, t, last_state, action, reward, state):
        self._learn(0, last_state, self._last_action, action, reward, state)
        self._last_action = action

    def _learn(self, t, last_state, last_action, action, reward, state):
        scalar_reward = np.dot(self._scalarization_weights.T, reward)
        self._Q[last_state, last_action] += self._alpha * (scalar_reward +
                                                           self._gamma *
                                                           self._Q[state, action] -
                                                           self._Q[last_state, last_action])
        if my_debug:
            log.debug(' Q: %s' % (str(self._Q)))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            action = random.choice(np.where(self._Q[state, :] == max(self._Q[state, :]))[0])
            # action = self._Q[state, :].argmax()
            if my_debug:
                log.debug('  took greedy action %i' % (action))
            return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        if my_debug:
            log.debug('   took random action %i' % (action))
        return action

    def get_learned_action(self, state):
        return self._Q[state, :].argmax()

    def get_learned_action_gibbs_distribution(self, state):
        tau = 2.0
        tmp = np.exp(self._Q[state, :] / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()

    def get_learned_action_distribution(self, state):
        tau = 0.6
        tmp = np.exp(np.dot(self._Q[state, :], self._scalarization_weights) / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()

    def reset(self):
        """
        resets the current agent! Be careful and save learned Q matrix first!
        """
        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)

    def save(self):
        self._Q_save.append(self._Q)

    def restore(self):
        tmp = np.zeros_like(self._Q)
        for i in self._Q_save:
            tmp += i
        self._Q = np.divide(tmp, self._Q_save.__len__())


class SARSALambdaMorlAgent(SARSAMorlAgent):
    """
    SARSA with eligibility traces.
    """
    def __init__(self, problem, scalarization_weights, alpha=0.3, epsilon=1.0, lmbda=0.7, **kwargs):
        super(SARSALambdaMorlAgent, self).__init__(problem, scalarization_weights, alpha, epsilon, **kwargs)

        self._lmbda = lmbda
        self._e = np.zeros_like(self._Q)
        # hidden variables for conserving agent state
        self._e_save = []

    def name(self):
        return "SARSALambda_" + str(self._lmbda) + "e" + str(self._epsilon) + "a" + str(self._alpha) + "W=" +\
               self._scalarization_weights.ravel().tolist().__str__()

    def _learn(self, t, last_state, last_action, action, reward, state):
        scalar_reward = np.dot(self._scalarization_weights.T, reward)
        delta = scalar_reward + self._gamma * self._Q[state, action] - self._Q[last_state, action]
        self._e[last_state, action] = min(self._e[last_state, action] + 1.0, 2.0)
        self._Q += self._alpha * delta * self._e
        self._e *= self._gamma * self._lmbda
        if my_debug: log.debug(' Q: %s' % (str(self._Q)))

    def reset(self):
        """
        resets the current agent! Be careful and save learned Q matrix and the eligibility traces first!
        """
        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)
        self._e = np.zeros_like(self._Q)

    def save(self):
        self._Q_save.append(self._Q)
        self._e_save.append(self._e)

    def restore(self):
        tmp = np.zeros_like(self._Q)
        for i in self._Q_save:
            tmp += i
        self._Q = np.divide(tmp, self._Q_save.__len__())
        tmp = np.zeros_like(self._Q)
        for i in self._e_save:
            tmp += i
        self._e = np.divide(tmp, self._e_save.__len__())


class QMorlAgent(MorlAgent):
    """
    A MORL agent, that uses Q learning.
    """

    def __init__(self, problem, scalarization_weights, alpha=0.3, epsilon=1.0, **kwargs):
        """
        Initialize the Reinforcement Learning MORL
        Agent with the problem description and alpha,
        the learning rate.

        Parameters
        ----------
        :param problem: A MORL problem
        :param scalarization_weights: a weight vector to scalarize the morl reward.
        :param alpha: real, the learning rate in each
            Q update step
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(QMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._epsilon = epsilon
        # the Q function is only one dimensional, since
        # we can only be in one state and choose from
        # two actions in the newcomb problem.
        self._Q = np.zeros(
                (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        # self._Q = np.ones(
        #         (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)

        # hidden variables for conserving agent state
        self._Q_save = []

    def name(self):
        return "scalQ_e" + str(self._epsilon) + "a" + str(self._alpha) + "W=" +\
               self._scalarization_weights.ravel().tolist().__str__()

    def learn(self, t, last_state, action, reward, state):
        self._learn(0, last_state, action, reward, state)
        self._last_action = action

    def _learn(self, t, last_state, action, reward, state):
        """
        Updating the Q-table according to Suttons Q-learning update for multiple
        objectives
        :return:
        """

        # scalar_reward = np.dot(self._scalarization_weights.T, reward)

        self._Q[last_state, action] += self._alpha *\
            (reward + self._gamma * np.amax(self._Q[state, :], axis=0) - self._Q[last_state, action])

        if my_debug:
            log.debug(' Q: %s' % (str(self._Q[state, :, :])))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            weighted_q = np.dot(self._Q[state, :], self._scalarization_weights)
            action = random.choice(np.where(weighted_q == max(weighted_q))[0])

            if my_debug:
                log.debug('  took greedy action %i' % action)
            return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        if my_debug:
            log.debug('   took random action %i' % action)
        return action

    def get_learned_action(self, state):
        return np.dot(self._Q[state, :], self._scalarization_weights).argmax()

    def get_learned_action_gibbs_distribution(self, state):
        tau = 0.6
        tmp = np.exp(np.dot(self._Q[state, :], self._scalarization_weights) / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()
        # action_value = np.dot(self._Q[state, :], self._scalarization_weights)
        # action_value = action_value.ravel()
        # action_value /= action_value.sum()
        # return action_value

    def get_learned_action_distribution(self, state):
        tau = 0.2
        tmp = np.exp(np.dot(self._Q[state, :], self._scalarization_weights) / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()

    def reset(self):
        """
        resets the current agent! Be careful and save learned Q matrix first!
        """
        self._Q = np.zeros(
                (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension))
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)

    def save(self):
        self._Q_save.append(self._Q)

    def restore(self):
        tmp = np.zeros(self._Q.shape)
        for i in self._Q_save:
            tmp += i
        self._Q = np.divide(tmp, self._Q_save.__len__())


class PreScalarizedQMorlAgent(MorlAgent):
    """
    A MORL agent, that uses Q learning with a scalar
    value function, scalarizing on every learning step
    """

    def __init__(self, problem, scalarization_weights, alpha=0.3, epsilon=1.0, **kwargs):
        """
        Initialize the Reinforcement Learning MORL
        Agent with the problem description and alpha,
        the learning rate.

        Parameters
        ----------
        :param problem: A MORL problem
        :param scalarization_weights: a weight vector to scalarize the morl reward.
        :param alpha: real, the learning rate in each
            Q update step
        :param epsilon: real, [0, 1] the epsilon factor for
            the epsilon greedy action selection strategy
        """
        super(PreScalarizedQMorlAgent, self).__init__(problem, **kwargs)

        self._scalarization_weights = scalarization_weights
        self._alpha = alpha
        self._epsilon = epsilon

        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._Q_save = []
        self._last_action = random.randint(0, problem.n_actions-1)

    def name(self):
        return "preScalQ_e" + str(self._epsilon) + "a" + str(self._alpha) + "W=" +\
               self._scalarization_weights.ravel().tolist().__str__()

    def learn(self, t, last_state, action, reward, state):
        # self._learn(0, last_state, self._last_action, reward, state)
        self._learn(0, last_state, action, reward, state)
        # Update last action after learning
        self._last_action = action

    def _learn(self, t, last_state, action, reward, state):
        """
        Updating the Q-table according to Suttons Q-learning update for multiple
        objectives
        :return:
        """
        scalar_reward = np.dot(self._scalarization_weights.T, reward)

        self._Q[last_state, action] += self._alpha * \
            (scalar_reward + self._gamma * np.amax(self._Q[state, :]) - self._Q[last_state, action])

        if my_debug:
            log.debug(' Q: %s' % (str(self._Q[state, :])))

    def decide(self, t, state):
        if random.random() < self._epsilon:
            action = random.choice(np.where(self._Q[state, :] == np.amax(self._Q[state, :]))[0])

            if my_debug:
                log.debug('  took greedy action %i' % action)
            return action
        action = random.randint(0, self._morl_problem.n_actions - 1)
        if my_debug:
            log.debug('   took random action %i' % action)
        return action

    def get_learned_action(self, state):
        return self._Q[state, :].argmax()

    def get_learned_action_gibbs_distribution(self, state):
        tau = 1.0
        tmp = np.exp(self._Q[state, :] / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()

    def get_learned_action_distribution(self, state):
        tau = 0.1
        tmp = np.exp(self._Q[state, :] / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()

    def reset(self):
        """
        resets the current agent! Be careful and save learned Q matrix first!
        """
        self._Q = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)

    def save(self):
        self._Q_save.append(self._Q)

    def restore(self):
        tmp = np.zeros(self._Q.shape)
        for i in self._Q_save:
            tmp += i
        self._Q = np.divide(tmp, self._Q_save.__len__())


class FixedPolicyAgent(MorlAgent):
    """
    An agent, that follows a fixed policy, defined as a morl policy object.
    No learning implemented.
    """
    def __init__(self, morl_problem, policy, **kwargs):
        super(FixedPolicyAgent, self).__init__(morl_problem, **kwargs)

        self._policy = policy

    def learn(self, t, last_state, action, reward, state):
        pass

    def decide(self, t, state):
        return self._policy.decide(state)

    def reset(self):
        """
        resets the current agent!
        """
        pass

    def save(self):
        pass

    def restore(self):
        pass

    def get_learned_action_gibbs_distribution(self, state):
        return self._policy._pi[state, :]


class NFQAgent(MorlAgent):
    """
    Implements neural fitted-Q iteration

    Can be used with scenarios where the reward vector contains only
    values between -1 and 1.

    TODO: Currently only morl problems with a cartesian coordinate
    system as state can be used.
    """
    def __init__(self, morl_problem, scalarization_weights, gamma, epsilon, **kwargs):
        super(NFQAgent, self).__init__(morl_problem, **kwargs)

        self._gamma = gamma
        self._epsilon = epsilon
        self._scalarization_weights = scalarization_weights

        self._transistion_history = []  # full transition history (s,a,a')
        self._train_history = []  # input history for NN (s,a)
        self._goal_hist = []  # goal history
        self._last_state = 0
        self._last_action = random.randint(0, morl_problem.n_actions-1)
        self._last_reward = np.zeros_like(self._scalarization_weights)

        # Create network with 2 layers and random initialized
        # self._net = nl.net.newff([[0, self._morl_problem.n_states], [0, 3]], [20, 20, 1])
        self._net = nl.net.newff([[0, self._morl_problem.scene_y_dim],
                                  [0, self._morl_problem.scene_x_dim],
                                  [0, self._morl_problem.n_actions]],
                                 [20, 20, len(self._scalarization_weights)])

    def reset(self):
        self._last_state = 0
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)
        self._transistion_history = []  # full transition history (s,a,a')
        self._train_history = []  # input history for NN (s,a)
        self._goal_hist = []  # goal history
        self._last_reward = np.zeros_like(self._scalarization_weights)
        self._net = nl.net.newff([[0, self._morl_problem.scene_y_dim],
                                  [0, self._morl_problem.scene_x_dim],
                                  [0, self._morl_problem.n_actions]],
                                 [20, 20, len(self._scalarization_weights)])

    def learn(self, t, last_state, action, reward, state):
        self._learn(0, self._last_state, self._last_action,
                    self._last_reward, action, reward, state)
        # state transition and update
        self._last_state = state
        self._last_action = action
        self._last_reward = reward

    def _learn(self, t, last_state, last_action, last_reward, action, reward, state):
        self._transistion_history.append([last_state, action, state])

        # Generate training set
        # self._train_history.append([last_state, last_action])
        self._train_history.append(np.hstack([np.array(self._morl_problem._get_position(last_state)), last_action]))

        Q_vals = []
        for i in xrange(self._morl_problem.n_actions):
            # Simulate network
            tmp = np.hstack([np.array(self._morl_problem._get_position(state)), i])
            Q_vals.append(self._net.sim(np.array([tmp])))

        tar_tmp = (reward + self._gamma * np.amax(Q_vals, axis=0))
        self._goal_hist.append(tar_tmp)

        # cost function (minimum time controller)
        # costs = 0.01
        # if self._morl_problem.terminal_state:
        #     costs += self._gamma * (1.0/reward[0])
        #     self._goal_hist.append(costs)
        #     if my_debug: log.debug('Terminal cost value in state %i is %f', state, costs)
        # else:
        #     Q_vals = []
        #     for i in xrange(self._morl_problem.n_actions):
        #         # Simulate network
        #         # Q_vals.append(self._net.sim(np.asarray([[state, i]])))
        #         tmp = np.hstack([np.array(self._morl_problem._get_position(state)), i])
        #         Q_vals.append(self._net.sim(np.array([tmp])))
        #
        #     costs += self._gamma * np.array(Q_vals).min()
        #     self._goal_hist.append(costs)
        #     if my_debug: log.debug('State cost value in state %i is %f', state, costs)

        inp = np.array(self._train_history)
        tar = np.array(self._goal_hist)
        # tar = tar.reshape(len(tar), 1)
        tar = tar.reshape(len(tar), self._morl_problem.reward_dimension)
        # Train network
        # error = self._net.train.train_rprop(input, target, epochs=500, show=100, goal=0.02)
        nl.train.train_rprop(self._net, inp, tar, epochs=500, show=0, goal=0.02)

        # Reset histories after termination of one episode
        if self._morl_problem.terminal_state:
            self._train_history = []  # input history for NN (s,a)
            self._goal_hist = []  # goal history

    def decide(self, t, state):
        # epsilon greedy
        if random.random() < self._epsilon:
            Q_vals = []
            for i in xrange(self._morl_problem.n_actions):
                # Simulate network
                # Q_vals.append(self._net.sim(np.asarray([[state, i]])))
                tmp = np.hstack([np.array(self._morl_problem._get_position(state)), i])
                Q_vals.append(self._net.sim(np.array([tmp])))

            weighted_q = np.dot(Q_vals, self._scalarization_weights)
            action = random.choice(np.where(weighted_q == max(weighted_q))[0])

            # action = random.choice(np.where(np.array(Q_vals) == min(np.array(Q_vals)))[0])
        else:
            action = random.randint(0, self._morl_problem.n_actions - 1)

        if my_debug:
            log.debug('Decided for action %i in state %i.', action, state)

        return action


class MORLScalarizingAgent(MorlAgent):
    """
    This class is an Agent that uses chebyshev scalarization method in Q-iteration
    Contains a Q-Value table with additional parameter o <-- (Objective)
    @author: Simon Wölzmüller <ga35voz@mytum.de>
    """
    def __init__(self, morl_problem, scalarization_weights, alpha, epsilon, tau, ref_point, function='chebishev',
                 gamma=0.9, **kwargs):
        """
        initializes MORL Agent
        :param morl_problem: a Problem inheriting MORLProblem Class
        :param scalarization_weights: vector for weighted sum of q values, weight on important objectives
        :param alpha: learning rate
        :param epsilon: for epsilon greedy action selection
        :param tau: small constant addition for z point, that is used as reference and optimized during learning
        :param kwargs: some additional arguments
        :return:
        """
        super(MORLScalarizingAgent, self).__init__(morl_problem, **kwargs)
        # create Q-value table
        self.q_shape = (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension)
        self._Q = np.zeros(self.q_shape)
        # parameter for Q-learning algorithm
        self._gamma = gamma
        self._alpha = alpha
        # parameter for greedy strategy
        self._epsilon = epsilon
        # small constant training addition value for the z point
        self._tau = tau
        # determine scalarization function:
        self.function = function
        if self.function == 'chebishev':
            # create reference point for each objective used for chebyshev scalarization adapted on each step
            self._z = np.zeros(self._morl_problem.reward_dimension)

        # weight vector
        self._w = scalarization_weights
        # last action is stored
        self._last_action = random.randint(0, morl_problem.n_actions-1)
        # hypervolume calculator
        self.hv_calculator = HyperVolumeCalculator(ref_point)
        # stores q values
        self.l = []
        # storage for volumes per interaction
        self.max_volumes = []
        # storage for the volumes optained in ONE interaction
        self.temp_vol = []
        self._Q_save = []

    def reset(self):
        self._z = np.zeros(self._morl_problem.reward_dimension)
        self._Q = np.zeros(self.q_shape)
        self.max_volumes = []
        self.l = []
        self.temp_vol = []

    def save(self):
        self._Q_save.append(self._Q)

    def restore(self):
        tmp = np.zeros_like(self._Q)
        for i in self._Q_save:
            tmp += i
        self._Q = np.divide(tmp, self._Q_save.__len__())

    def decide(self, t, state):
        """
        This function decides using a epsilon greedy algorithm and chebishev scalarizing function.
        :param state: the state the agent is at the moment
        :param t: iteration
        :return: action the agent chose
        """
        # epsilon greedy action selection:
        if random.random() > self._epsilon:
            # greedy action selection:
            action = self._greedy_sel(state)
        else:
            # otherwise choose randomly over action space
            action = random.randint(0, self._morl_problem.n_actions - 1)
        if my_debug:
            # log the decided action
            log.debug('Decided for action %i in state %i.', action, state)
        return action

    def learn(self, t, last_state, action, reward, state):
        # access private function
        self._learn(t, last_state, action, reward, state)
        #  store last action after learning
        self._last_action = action

    def _learn(self, t, last_state, action, reward, state):
        """
        learn like they do in Van Moeffart/Drugan/Nowé paper about scalarization
        :return:
        """
        new_action = self._greedy_sel(state)
        # update rule for every objective
        for objective in xrange(self._morl_problem.reward_dimension):
            # advanced bellman equation
            self._Q[last_state, action, objective] += self._alpha * \
                (reward[objective] + self._gamma * self._Q[state, new_action, objective] -
                 self._Q[last_state, action, objective])
            # store z value
            if self.function =='chebishev':
                self._z[objective] = self._Q[:, :, objective].max() + self._tau

    def get_learned_action(self, state):
        """
        uses epsilon greedy and weighted scalarisation for action selection
        :param state: state the agent is
        :return: action to do next
        """
        #  state -> quality list
        sq_list = []
        # explore all actions
        for acts in xrange(self._morl_problem.n_actions):
            # create value vector for objectives
            obj = [x for x in self._Q[state, acts, :]]
            if self.function == 'linear':
                sq = sum([(self._w[o]*obj[o]) for o in xrange(len(obj))])
            if self.function == 'chebishev':
                sq = np.amax([self._w[o]*abs(obj[o]-self._z[o]) for o in xrange(len(obj))])
            # store that value into the list
            sq_list.append(sq)
        # chosen action is the one with greatest sq value
        if self.function == 'linear':
            new_action = np.argmax(sq_list)
        if self.function == 'chebishev':
            new_action = np.argmin(sq_list)


        return new_action


    def get_learned_action_gibbs_distribution(self, state):
        """
        uses gibbs distribution to decide which action to do next
        :param state: given state the agent is atm
        :return: an array of actions to do next, with probability
        """
        tau = 0.6
        tmp = np.exp(np.dot(self._Q[state, :], self._w) / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()

    def name(self):
        return "skalar"+self.function+"_Q_agent_e" + str(self._epsilon) + "a" + str(self._alpha) + "W=" + str(self._w)

    def _greedy_sel(self, state):
        # state -> quality list
        sq_list = []
        # explore all actions
        for acts in xrange(self._morl_problem.n_actions):
            # create value vector for objectives
            obj = [x for x in self._Q[state, acts, :]]
            if self.function == 'linear':
                sq = sum([(self._w[o]*obj[o]) for o in xrange(len(obj))])
            if self.function == 'chebishev':
                sq = np.amax([self._w[o]*abs(obj[o]-self._z[o]) for o in xrange(len(obj))])
            # store that value into the list
            sq_list.append(sq)
        # chosen action is the one with greatest sq value
        if self.function == 'linear':
            new_action = np.argmax(sq_list)
        if self.function == 'chebishev':
            new_action = np.argmin(sq_list)
        q = np.array(([x for x in self._Q[state, new_action, :]]))
        self.l.append(q)
        # store hv, only if self.l is non-empty (only this way worked for me TODO: find elegant way )
        if len(self.l):
            # catch the list
            l = np.array(self.l)
            # only the points on front are needed
            l = self.hv_calculator.extract_front(l)
            # restore it into the member
            self.l = [x for x in l]
            # compute new hypervolume:
            self.temp_vol.append(self.hv_calculator.compute_hv(self.l))
            self.temp_vol = remove_duplicates(self.temp_vol)
            # at the end of an interaction:
            if self._morl_problem.terminal_state:
                # store the maximum hypervolume
                self.max_volumes.append(max(self.temp_vol))
                # clear the temporary list
                self.temp_vol = []
        return new_action


class MORLHVBAgent(MorlAgent):
    """
    this class is implemenation of hypervolume based MORL agent,
    the reference point (ref) is used for quality evaluation of
    state-action lists depending on problem set.
    @author: Simon Wölzmüller <ga35voz@mytum.de>
    """
    def __init__(self, morl_problem, alpha, epsilon, ref, scal_weights,  **kwargs):
        """
        initializes agent with params used for Q-learning and greedy decision
        :param morl_problem:
        :param alpha: learning rate for q learning
        :param: epsilon: probability of epsilon greedy algorithm
        :param: ref: reference point for hypervolume calculation
        :return:
        """
        super(MORLHVBAgent, self).__init__(morl_problem, **kwargs)
        # create Q-value table
        self.q_shape = (self._morl_problem.n_states, self._morl_problem.n_actions, self._morl_problem.reward_dimension)
        self._Q = np.zeros(self.q_shape)
        # self._Q = np.random.randint(0, 100, self.q_shape)

        # learning rate for Q-learning algorithm
        self._alpha = alpha
        # parameter for epsilon greedy strategy
        self._epsilon = epsilon
        # hv calculator
        self.hv_calculator = HyperVolumeCalculator(ref)
        # init last action
        self._last_action = random.randint(0, morl_problem.n_actions-1)
        # storage for temporal volumes one for each interaction
        self.temp_vol = []
        # storage for max volumes
        self.max_volumes = []
        # storage for q values
        self._l = []
        # weights for objectives
        self._w = scal_weights
        self._Qsave = []

    def save(self):
        self._Qsave.append(self._Q)

    def restore(self):
        tmp = np.zeros_like(self._Q)
        for i in self._Q_save:
            tmp += i
        self._Q = np.divide(tmp, self._Q_save.__len__())

    def reset(self):
        self._Q = np.zeros(self.q_shape)
        self._last_action = random.randint(0, self._morl_problem.n_actions-1)
        self.max_volumes =[]
        self._l = []
        self.temp_vol =[]

    def decide(self, t, state):
        # epsilon greedy hypervolume based action selection:
        if random.random() > self._epsilon:
            # greedy action selection:
            action = self._greedy_sel(state)
        else:
            # otherwise choose randomly over action space
            action = random.randint(0, self._morl_problem.n_actions - 1)

        if my_debug:
            # log the decided action
            log.debug('Decided for action %i in state %i.', action, state)

        return action

    def learn(self, t, last_state, action, reward, state):
        """
        this function is an adaption of the hvb q learning algorithm from van moeffart/drugan/nowé
        :param t: count of iterations
        :param last_state: last state before transition to this state
        :param action: action to chose after this state found by HVBgreedy
        :param reward: reward received from this action for every objective
        :param state: state we're currently being in
        :return:
        """
        # access private function
        self._learn(t, last_state, action, reward, state)
        #  store last action after learning
        self._last_action = action

    def _learn(self, t, last_state, action, reward, state):
        # append new q values to list
        self._l.append(np.array([x for x in self._Q[state, action, :]]))
        # if there isnt any value yet in the list, dont calculate...
        if len(self._l):
            # create numpy array, needed by the hv calculator
            l = np.array(self._l)
            # compute
            l = self.hv_calculator.extract_front(l)
            # recreate a python list, to allow easier appending
            self._l = [x for x in l]
        # decide which state we're up to take
        new_action = self._greedy_sel(state)
        # update q values
        for objective in xrange(self._morl_problem.reward_dimension):
            # advanced bellman equation
            self._Q[last_state, action, objective] += self._alpha * \
                (reward[objective] + self._gamma * self._Q[state, new_action, objective] -
                 self._Q[last_state, action, objective])

    def _greedy_sel(self, state):
        """
        action selection strategy based on hypervolume indicator
        :param state:
        :return:
        """
        volumes = []
        for act in xrange(self._morl_problem.n_actions):
            # store l list in local copy
            l_set = []
            # append list on local copy
            if len(self._l):
                l_set.append(self._l)
            # append new opjective vector
            l_set.append(self._Q[state, act, :])
            # append to other volumes
            volumes.append(self.hv_calculator.compute_hv(l_set))
        # best action has biggest hypervolume
        new_action = volumes.index(max(volumes))
        self.temp_vol.append(max(volumes))
        # if the interaction has finished
        if self._morl_problem.terminal_state:
            # append the maximum volume to the master volume list
            self.max_volumes.append(max(self.temp_vol))
            # clear the temporary list
            self.temp_vol = []
        return new_action

    def name(self):
        return "HVB_Q_agent_" + 'e' + str(self._epsilon) + "a" + str(self._alpha)

    def get_learned_action_distribution(self, state):
        """
        scalarized greedy action selection
        :param state: state we're being in
        :return:
        """
        if random.random() < self._epsilon:
            # dot product with weights
            weighted_q = np.dot(self._Q[state, :], self._w)
            # the first of maximum list (if more are available)
            action = random.choice(np.where(weighted_q == max(weighted_q))[0])
            return action
        else:
            return random.randint(0, self._morl_problem.n_actions-1)

    def get_learned_action(self, state):
        """
        uses epsilon greedy and hvb action selection
        :param state: state the agent is
        :return: action to do next
        """
        return self._greedy_sel(state)


    def get_learned_action_gibbs_distribution(self, state):
        """
        uses gibbs distribution to decide which action to do next
        :param state: given state the agent is atm
        :return: an array of actions to do next, with probability
        """
        tau = 0.6
        tmp = np.exp(np.dot(self._Q[state, :], self._w) / tau)
        tsum = tmp.sum()
        dist = tmp / tsum
        return dist.ravel()


class MORLHLearningAgent(MorlAgent):
    def __init__(self, morl_problem, epsilon, alpha, weights, **kwargs):
        super(MORLHLearningAgent, self).__init__(morl_problem, **kwargs)
        # probability for epsilon greedy
        self._epsilon = epsilon
        # weight vector
        self.w = weights
        # count of each action an state is left with
        self.n_action_taken = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions))
        self.n_action_taken[:, :] = self._morl_problem.n_states
        # count of results an action ends with
        self.n_action_resulted_in = np.ones((self._morl_problem.n_states, self._morl_problem.n_actions,
                                             self._morl_problem.n_states))
        # probability of an state action state triple
        self._probability = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions,
                                      self._morl_problem.n_states))
        self._probability[:, :, :] = 1.0/float(self._morl_problem.n_states)
        # reward for state-state transition
        self._reward = np.zeros((self._morl_problem.n_states, self._morl_problem.n_states,
                                 self._morl_problem.reward_dimension))
        # h value
        self._h = np.zeros((self._morl_problem.n_states, self._morl_problem.reward_dimension))
        # scalar optimal average reward
        self._rho = np.zeros(self._morl_problem.reward_dimension)
        self.alpha = alpha
        self.took_greedy = False

    def decide(self, t, state):
        if random.random() < self._epsilon:
            return random.randint(0, self._morl_problem.n_actions-1)
        else:
            return self._greedy_sel(t, state)

    def _greedy_sel(self, t, state):
        a_list = []
        for action in xrange(self._morl_problem.n_actions):
            suma = np.sum([self._probability[state, action, i]*self._h[i] for i in range(self._morl_problem.n_states)])

            a_list.append(np.dot((self._reward[state, action] + suma), self.w))
        self.took_greedy = True
        return a_list.index(max(a_list))

    def learn(self, t, last_state, action, reward, state):
        # access private function
        self._learn(t, last_state, action, reward, state)
        #  store last action after learning
        self._last_action = action

    def _learn(self, t, last_state, action, reward, state):
        # count up action taken from last state
        self.n_action_taken[last_state, action] += 1
        # count up state resulted from this state action
        self.n_action_resulted_in[last_state, action, state] += 1
        self._probability[last_state, action, state] = self.n_action_resulted_in[last_state, action, state] / \
            self.n_action_taken[last_state, action]
        self._reward[last_state, action] += (reward - self._reward[last_state, action]) / \
            self.n_action_taken[last_state, action]
        if self.took_greedy:
            self.took_greedy = False
            self._rho = self.alpha*(reward - self._h[last_state] + self._h[state]) + self._rho * (1-self.alpha)
            self.alpha = (self.alpha / (1 + self.alpha))
        self._get_h_value(last_state, action)

    def _get_h_value(self, state, action):
        h_list = []
        h_list.append(self.w*(self._reward[state, action] + np.sum([self._probability[state, action, next_state] *
                                                                    self._h[next_state, :]
                                                                    for next_state
                                                                    in xrange(self._morl_problem.n_states)], axis=0)))
        self._h[state] = max(h_list) - self._rho

    def get_learned_action(self, state):
        """
        uses epsilon greedy and h action selection
        :param state: state the agent is
        :return: action to do next
        """
        # get action out of max q value of n_objective-dimensional matrix
        temp = self._epsilon
        self._epsilon = 1
        if random.random() < self._epsilon:
            self._epsilon = temp
            return self._greedy_sel(0, state)

        else:
            return random.randint(0, self._morl_problem.n_actions-1)


class MORLRLearningAgent(MorlAgent):
    def __init__(self, morl_problem, epsilon, alpha, beta, weights, **kwargs):
        super(MORLRLearningAgent, self).__init__(morl_problem, **kwargs)
        # probability for epsilon greedy
        self._epsilon = epsilon
        # weight vector
        self.w = weights
        self._beta = beta
        # reward for state-state transition
        self._reward = np.zeros((self._morl_problem.n_states, self._morl_problem.n_states,
                                 self._morl_problem.reward_dimension))
        # R value
        self._R = np.zeros((self._morl_problem.n_states, self._morl_problem.n_actions,
                            self._morl_problem.reward_dimension))
        # scalar optimal average reward
        self._rho = np.zeros(self._morl_problem.reward_dimension)
        # control for "best action token"
        self.alpha = alpha
        self.took_greedy = False

    def decide(self, t, state):
        if random.random() < self._epsilon:
            return random.randint(0, self._morl_problem.n_actions-1)
        else:
            return self._greedy_sel(t, state)

    def _greedy_sel(self, t, state):
        a_list = []
        for action in xrange(self._morl_problem.n_actions):
            a_list.append(np.dot(self._R[state, action, :], self.w))
        self.took_greedy = True
        return a_list.index(max(a_list))

    def learn(self, t, last_state, action, reward, state):
        # access private function
        self._learn(t, last_state, action, reward, state)
        #  store last action after learning
        self._last_action = action

    def _learn(self, t, last_state, action, reward, state):
        rew = [np.dot(self._R[state, a, :], self.w) for a in xrange(self._morl_problem.n_actions)]
        maxa = rew.index(max(rew))
        self._R[last_state, action, :] = self._R[last_state, action]*(1-self._beta) +\
            self._beta * (reward - self._rho + self._R[state, maxa, :])
        if self.took_greedy:
            self.took_greedy = False

            self._rho = self.alpha * (reward - self._R[last_state, action, :] + self._R[state, maxa, :]) +\
                self._rho * (1-self.alpha)
            self.alpha = (self.alpha / (0.001 + self.alpha))

    def get_learned_action(self, state):
        """
        uses epsilon greedy and hvb action selection
        :param state: state the agent is
        :return: action to do next
        """
        # get action out of max q value of n_objective-dimensional matrix
        if random.random() < self._epsilon:
            return self._greedy_sel(0, state)
        else:
            return random.randint(0, self._morl_problem.n_actions-1)

