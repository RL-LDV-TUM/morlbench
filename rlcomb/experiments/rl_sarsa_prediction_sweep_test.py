'''
Created on Nov 21, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''


'''
Experiment that sweeps over the prediction accuracy of the
Newcomb problems predictor and does a RL learning over 
10000 iterations with SARSA.
'''

import sys
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import matplotlib.pyplot as plt

log.basicConfig(level=log.INFO)

from problems import Newcomb
from agents import RLNewcombAgent


if __name__ == '__main__':
    independent_runs = 200
    interactions = 1000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 50

    avg_payouts = np.zeros((independent_runs, linspace_steps))
    learned_actions = np.zeros((independent_runs, linspace_steps))

    for r in xrange(independent_runs):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for prediction_accuracy in np.linspace(linspace_from, linspace_to,
                                               linspace_steps):
            problem = Newcomb(predictor_accuracy=prediction_accuracy,
                              payouts=np.array([[1000000, 0],
                                                [1001000, 1000]]))
            agent = RLNewcombAgent(problem, alpha=0.3, gamma=0.9, epsilon=0.8)

            log.info('Playing ...')
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            payouts = agent.interact(interactions)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            log.info('Average Payout for predicion accuraccy %.3f: %.3f' % \
                     (prediction_accuracy, avg_payout))

            learned_actions_in_run.append(agent.get_learned_action())

        avg_payouts[r, :] = np.array(avg_payouts_in_run)
        learned_actions[r, :] = np.array(learned_actions_in_run)

    avg_payouts = avg_payouts.mean(axis=0)
    learned_actions = learned_actions.mean(axis=0)

    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('payout')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), avg_payouts, label='RLAgent')
    plt.legend(loc='upper center')
    plt.savefig("rl_agent_payout.png")
    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('learned action')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), learned_actions, label='RLAgent')
    plt.savefig("rl_agent_learned_action.png")
