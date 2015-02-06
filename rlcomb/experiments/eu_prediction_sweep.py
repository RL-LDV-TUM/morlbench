'''
Created on Nov 21, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Experiment that sweeps over the prediction accuracy of the
Newcomb problems predictor and does a UCB1 learning over 
10000 iterations.
'''

import sys
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from problems import Newcomb
from agents import EUNewcombAgent


if __name__ == '__main__':
    independent_runs = 1
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    avg_payouts = np.zeros((independent_runs, linspace_steps))
    learned_actions = np.zeros((independent_runs, linspace_steps))

    def onerun(r):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for prediction_accuracy in np.linspace(linspace_from, linspace_to,
                                               linspace_steps):
            problem = Newcomb(predictor_accuracy=prediction_accuracy,
                              payouts=np.array([[1000000, 0],
                                                [1001000, 1000]]))
            agent = EUNewcombAgent(problem)

            log.info('Playing ...')
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            payouts = agent.interact_multiple(interactions)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            log.info('Average Payout for predicion accuraccy %.3f: %.3f' % \
                     (prediction_accuracy, avg_payout))

            learned_actions_in_run.append(agent.get_learned_action())

        return (np.array(avg_payouts_in_run), np.array(learned_actions_in_run))

    results = Parallel(n_jobs=1)(delayed(onerun)(r) for r in
                                 xrange(independent_runs))

    for r in xrange(len(results)):
        avg_payouts[r, :] = results[r][0]
        learned_actions[r, :] = results[r][1]

    avg_payouts = avg_payouts.mean(axis=0)
    learned_actions = learned_actions.mean(axis=0)

    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('payout')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), avg_payouts, label='EUAgent')
    plt.legend(loc='upper center')
    plt.savefig("eu_agent_payout.png")
    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('learned action')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), learned_actions, label='EUAgent')
    plt.savefig("eu_agent_learned_action.png")
