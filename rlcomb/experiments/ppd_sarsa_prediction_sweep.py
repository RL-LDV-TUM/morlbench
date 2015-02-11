'''
Created on Feb, 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''


'''
Experiment that sweeps over the prediction accuracy of the
Prisoner's Dilemma predictor and does learning over
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

from plotting_stuff import plot_that_pretty_rldm15

from problems import ProbabilisticPrisonersDilemma
from agents import SARSAPrisonerAgent


if __name__ == '__main__':
    independent_runs = 2
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 10

    avg_payouts = np.zeros((independent_runs, linspace_steps))
    learned_actions = np.zeros((independent_runs, linspace_steps))

    def onerun(r):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for cooperation_ratio in np.linspace(linspace_from, linspace_to,
                                             linspace_steps):
            problem = ProbabilisticPrisonersDilemma(coop_p=cooperation_ratio)
            agent = SARSAPrisonerAgent(problem)

            log.info('Playing ...')
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            payouts = []
            for _ in xrange(interactions):
                action = agent.decide()
                payout = problem.play(action)
                payouts.append(payout[0])
                agent.learn(action, payout[0])
            payouts = np.array(payouts)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            log.info('Average Payout for cooperation ratio %.3f: %.3f' %
                     (cooperation_ratio, avg_payout))

            learned_actions_in_run.append(agent.get_learned_action())

        return (np.array(avg_payouts_in_run), np.array(learned_actions_in_run))

    results = Parallel(n_jobs=-1)(delayed(onerun)(r) for r in
                                  xrange(independent_runs))

    for r in xrange(len(results)):
        avg_payouts[r, :] = results[r][0]
        learned_actions[r, :] = results[r][1]

    avg_payouts = avg_payouts.mean(axis=0)
    learned_actions = learned_actions.mean(axis=0)

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [learned_actions],
                            ["SARSA Agent"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Learned Action",
                            (0, 1.1, 0.2),
                            'sarsa_pd_payout.pdf')

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [learned_actions],
                            ["SARSA Agent"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Learned Action",
                            (0, 1.1, 0.2),
                            'sarsa_pd_learned_action.pdf')

#     fig = plt.figure()
#     plt.xlabel('prediction accuracy')
#     plt.ylabel('payout')
#     plt.plot(np.linspace(linspace_from, linspace_to,
#                          linspace_steps), avg_payouts,
#              label='SARSAPrisonerAgent')
#     plt.legend(loc='upper center')
#     plt.savefig("sarsa_pd_payout.png")
#     fig = plt.figure()
#     plt.xlabel('prediction accuracy')
#     plt.ylabel('learned action')
#     plt.plot(np.linspace(linspace_from, linspace_to,
#                          linspace_steps), learned_actions,
#              label='SARSAPrisonerAgent')
#     plt.savefig("sarsa_pd_learned_action.png")
