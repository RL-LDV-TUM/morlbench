'''
Created on Apr 23, 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

import sys
import os
sys.path.append(os.path.join('..', '..'))
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from problems import PrisonersDilemma
from agents import SARSAPrisonerAgent
from experiments.experiment_helpers import interact_multiple_twoplayer


if __name__ == '__main__':
    independent_runs = 1
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 2

    def onerun(r, pparams, use_total_payout):
        avg_payouts_in_run1 = []
        std_payouts_in_run1 = []
        avg_payouts_in_run2 = []
        std_payouts_in_run2 = []
        learned_actions_in_run1 = []
        learned_actions_in_run2 = []
        total_payouts_in_run = []

        for cooperation_ratio in np.linspace(linspace_from, linspace_to,
                                             linspace_steps):
            problem = PrisonersDilemma(**pparams)
            agent1 = SARSAPrisonerAgent(problem, alpha=0.1, gamma=0.2,
                                        epsilon=0.9)
            agent2 = SARSAPrisonerAgent(problem, alpha=0.1, gamma=0.2,
                                        epsilon=0.9)

            log.info('Playing ...')
            log.info('%s' % (str(agent1)))
            log.info('%s' % (str(problem)))
            log.info(' VERSUS')
            log.info('%s' % (str(agent2)))
            log.info('%s' % (str(problem)))

            (_, payouts1), (_, payouts2) = interact_multiple_twoplayer(agent1,
                                                                       agent2,
                                                                       problem,
                                                                       interactions)
            avg_payout1 = payouts1.mean(axis=0)
            std_payout1 = payouts1.std(axis=0)
            avg_payout2 = payouts2.mean(axis=0)
            std_payout2 = payouts2.std(axis=0)
            total = payouts1 + payouts2
            avg_total = total.mean(axis=0)

            avg_payouts_in_run1.append(avg_payout1)
            std_payouts_in_run1.append(std_payout1)
            avg_payouts_in_run2.append(avg_payout2)
            std_payouts_in_run2.append(std_payout2)
            total_payouts_in_run.append(avg_total)

            learned_actions_in_run1.append(agent1.get_learned_action())
            learned_actions_in_run2.append(agent2.get_learned_action())

            log.info('Average Payout: %.3f vs. %.3f (total: %.3f)' %
                     (avg_payout1, avg_payout2, avg_total))

        return ((np.array(avg_payouts_in_run1),
                 np.array(learned_actions_in_run1)),
                (np.array(avg_payouts_in_run2),
                 np.array(learned_actions_in_run2)))

    exps = [(False,
             {'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
             "pd_sarsa_local_prediction_sweep_normal.pickle"),
            (False,
             {'T': 1001000.0, 'R': 50000.0, 'P': 1000.0, 'S': 0.0},
             "pd_sarsa_local_prediction_sweep_modified.pickle")
            ]

    for use_total_payout, pparams, picklefile in exps:
        results = None

        if os.path.exists(picklefile):
            continue

        results = Parallel(n_jobs=1)(delayed(onerun)(r, pparams,
                                                     use_total_payout) for r in
                                     xrange(independent_runs))
        with open(picklefile, 'wb') as f:
            pickle.dump(results, f)
