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
import cPickle as pickle
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from plotting_stuff import plot_that_pretty_rldm15

from problems import ProbabilisticPrisonersDilemma
from agents import SARSAPrisonerAgent, AVGQPrisonerAgent


if __name__ == '__main__':
    independent_runs = 50
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100
#     linspace_from = 0.99
#     linspace_to = 0.99
#     linspace_steps = 1

    use_total_payout = True
    loadresults = False

    avg_payouts = np.zeros((independent_runs, linspace_steps))
    learned_actions = np.zeros((independent_runs, linspace_steps))

    def onerun(r):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for cooperation_ratio in np.linspace(linspace_from, linspace_to,
                                             linspace_steps):
            problem = ProbabilisticPrisonersDilemma(T=1001000.0, R=50000.0,
                                                    P=1000.0, S=0.0,
                                                    coop_p=cooperation_ratio)
#             problem = ProbabilisticPrisonersDilemma(coop_p=cooperation_ratio)
            agent = SARSAPrisonerAgent(problem, alpha=0.1, gamma=0.2,
                                       epsilon=0.9)

            log.info('Playing %i interactions...' % (interactions))
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            payouts = []
            for _ in xrange(interactions):
                action = agent.decide(0)
                payout = problem.play(action)
                if use_total_payout:
                    payout = payout[0] + payout[1]
                else:
                    payout = payout[0]
                payouts.append(payout)
                agent.learn(action, payout)
            payouts = np.array(payouts)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            log.info('Average Payout for cooperation ratio %.3f: %.3f' %
                     (cooperation_ratio, avg_payout))

            print agent.Q
            print agent.get_learned_action()

            learned_actions_in_run.append(agent.get_learned_action())

        return (np.array(avg_payouts_in_run), np.array(learned_actions_in_run))

    results = None

    if loadresults:
        with open("ppd_sarsa_prediction_sweep.pickle", 'rb') as f:
            results = pickle.load(f)
    else:
        results = Parallel(n_jobs=6)(delayed(onerun)(r) for r in
                                     xrange(independent_runs))
        with open("ppd_sarsa_prediction_sweep.pickle", 'wb') as f:
            pickle.dump(results, f)

    for r in xrange(len(results)):
        avg_payouts[r, :] = results[r][0]
        learned_actions[r, :] = results[r][1]

    avg_payouts = avg_payouts.mean(axis=0)
    learned_actions = learned_actions.mean(axis=0)

#     plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
#                                          linspace_steps)],
#                             [learned_actions],
#                             ["SARSA"],
#                             "Prediction Accuracy",
#                             (0, 1.1, 0.2),
#                             "Payout",
#                             (0, 1001000, 100000),
#                             'sarsa_pd_payout.pdf')
# 
#     plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
#                                          linspace_steps)],
#                             [learned_actions],
#                             ["SARSA"],
#                             "Prediction Accuracy",
#                             (0, 1.1, 0.2),
#                             "Learned Action",
#                             (0, 1.1, 0.2),
#                             'sarsa_pd_learned_action.pdf')

    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('payout')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), avg_payouts,
             label='SARSAPrisonerAgent')
    plt.legend(loc='upper center')
    plt.savefig("sarsa_pd_payout.pdf")
    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('learned action')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), learned_actions,
             label='SARSAPrisonerAgent')
    plt.savefig("sarsa_pd_learned_action.pdf")
