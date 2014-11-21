'''
Created on Nov 21, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''


'''
Experiment that sweeps number of episodes
of a RL (SARSA) Newcomb agent.
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
    independent_runs = 20
    prediction_accuracy = 0.2

    interactions_range = range(10, 10000, 100)

    learned_actions = np.zeros((independent_runs, len(interactions_range)))

    for r in xrange(independent_runs):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for interactions in interactions_range:
            problem = Newcomb(predictor_accuracy=prediction_accuracy,
                              payouts=np.array([[1000000, 0],
                                                [1001000, 1000]]))
            agent = RLNewcombAgent(problem, alpha=0.1, gamma=0.9, epsilon=0.9)

            log.info('Playing ...')
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            payouts = agent.interact(interactions)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            learned_action = agent.get_learned_action()

            log.info('Learned action %i with %i interaction episodes' %
                     (learned_action, interactions))

            learned_actions_in_run.append(learned_action)

        learned_actions[r, :] = np.array(learned_actions_in_run)

    learned_actions = learned_actions.mean(axis=0)

    plt.xlabel('interactions')
    plt.ylabel('learned action')
    plt.plot(np.array(interactions_range), learned_actions, label='RLAgent')
    plt.savefig("rl_agent_episode_sweep_learned_actions.png")
