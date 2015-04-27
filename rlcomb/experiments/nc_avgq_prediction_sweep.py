'''
Created on Nov 21, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Experiment that sweeps over the prediction accuracy of the
Prisoner's Dilemma problems predictor and does a RL learning over
10000 iterations with AVGQ
'''

import sys
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import cPickle as pickle
from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from plotting_stuff import plot_that_pretty_rldm15

from problems import Newcomb
from agents import AVGQNewcombAgent
from experiments.experiment_helpers import interact_multiple


if __name__ == '__main__':
    independent_runs = 50
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    loadresults = False

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
            agent = AVGQNewcombAgent(problem, alpha=0.1, gamma=0.9, epsilon=0.9)

            log.info('Playing ...')
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            _, payouts = interact_multiple(agent, problem, interactions)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            log.info('Average Payout for predicion accuraccy %.3f: %.3f' %
                     (prediction_accuracy, avg_payout))

            learned_actions_in_run.append(agent.get_learned_action())
        return (np.array(avg_payouts_in_run), np.array(learned_actions_in_run))

    if loadresults:
        with open("nc_avgq_prediction_sweep.pickle", 'rb') as f:
            results = pickle.load(f)
    else:
        results = Parallel(n_jobs=4)(delayed(onerun)(r) for r in
                                     xrange(independent_runs))
        with open("nc_avgq_prediction_sweep.pickle", 'wb') as f:
            pickle.dump(results, f)

    for r in xrange(len(results)):
        avg_payouts[r, :] = results[r][0]
        learned_actions[r, :] = results[r][1]

    avg_payouts = avg_payouts.mean(axis=0)
    learned_actions = learned_actions.mean(axis=0)

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts],
                            ["AVGQ"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Payout",
                            (0, 1001000, 100000),
                            'avgq_newcomb_payout.pdf')

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [learned_actions],
                            ["AVGQ"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Learned Action",
                            (0, 1.1, 0.2),
                            'avgq_newcomb_learned_action.pdf')

#     fig = plt.figure()
#     plt.xlabel('prediction accuracy')
#     plt.ylabel('payout')
#     plt.plot(np.linspace(linspace_from, linspace_to,
#                          linspace_steps), avg_payouts, label='RLAgent')
#     plt.legend(loc='upper center')
#     plt.savefig("rl_agent_payout.png")
#     fig = plt.figure()
#     plt.xlabel('prediction accuracy')
#     plt.ylabel('learned action')
#     plt.plot(np.linspace(linspace_from, linspace_to,
#                          linspace_steps), learned_actions, label='RLAgent')
#     plt.savefig("rl_agent_learned_action.png")
