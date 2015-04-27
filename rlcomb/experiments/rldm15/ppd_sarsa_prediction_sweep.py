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
import os
sys.path.append(os.path.join('..', '..'))
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from problems import ProbabilisticPrisonersDilemma
from agents import SARSAPrisonerAgent


if __name__ == '__main__':
    independent_runs = 50
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    def onerun(r, pparams, use_total_payout):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for cooperation_ratio in np.linspace(linspace_from, linspace_to,
                                             linspace_steps):
            #  problem = ProbabilisticPrisonersDilemma(T=1001000.0, R=50000.0,
            #                                          P=1000.0, S=0.0,
            #                                          coop_p=cooperation_ratio)
            #  problem = ProbabilisticPrisonersDilemma(coop_p=cooperation_ratio)
            pparams['coop_p'] = cooperation_ratio
            problem = ProbabilisticPrisonersDilemma(**pparams)
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

    exps = [(False,
             {'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
             "ppd_sarsa_local_prediction_sweep_normal.pickle"),
            (True,
             {'T': 5.0, 'R': 3.0, 'P': 1.0, 'S': 0.0},
             "ppd_sarsa_total_prediction_sweep_normal.pickle"),
            (False,
             {'T': 1001000.0, 'R': 50000.0, 'P': 1000.0, 'S': 0.0},
             "ppd_sarsa_local_prediction_sweep_modified.pickle"),
            (True,
             {'T': 1001000.0, 'R': 50000.0, 'P': 1000.0, 'S': 0.0},
             "ppd_sarsa_total_prediction_sweep_modified.pickle"),
            ]

    for use_total_payout, pparams, picklefile in exps:
        results = None

        if os.path.exists(picklefile):
            continue

        results = Parallel(n_jobs=-1)(delayed(onerun)(r, pparams,
                                                     use_total_payout) for r in
                                     xrange(independent_runs))
        with open(picklefile, 'wb') as f:
            pickle.dump(results, f)
