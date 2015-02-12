'''
Created on Feb 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Plot results from AVGQ and EU Newcomb agent experiments.
'''

import sys
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import os
import cPickle as pickle

log.basicConfig(level=log.INFO)

from plotting_stuff import plot_that_pretty_rldm15


if __name__ == '__main__':
    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    if not os.path.isfile("nc_avgq_prediction_sweep.pickle") or \
            not os.path.isfile("eu_prediction_sweep.pickle"):
        print >>sys.stderr, "run nc_avgq_prediction_sweep.py and eu_prediction_\
            sweep.py first to create the .pickle files"
        sys.exit(1)

    avgqresults = None
    euresults = None
    with open("nc_avgq_prediction_sweep.pickle", 'rb') as f:
        avgqresults = pickle.load(f)

    with open("eu_prediction_sweep.pickle", 'rb') as f:
        euresults = pickle.load(f)

    avg_payout_avgq = np.zeros((len(avgqresults), linspace_steps))
    learned_actions_avgq = np.zeros((len(avgqresults), linspace_steps))
    avg_payout_eu = np.zeros((len(euresults), linspace_steps))
    learned_actions_eu = np.zeros((len(euresults), linspace_steps))

    for r in xrange(len(avgqresults)):
        avg_payout_avgq[r, :] = avgqresults[r][0]
        learned_actions_avgq[r, :] = avgqresults[r][1]
    for r in xrange(len(euresults)):
        avg_payout_eu[r, :] = euresults[r][0]
        learned_actions_eu[r, :] = euresults[r][1]

    avg_payout_avgq = avg_payout_avgq.mean(axis=0)
    learned_actions_avgq = learned_actions_avgq.mean(axis=0)
    avg_payout_eu = avg_payout_eu.mean(axis=0)
    learned_actions_eu = learned_actions_eu.mean(axis=0)

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payout_avgq, avg_payout_eu],
                            ["AVGQ", "EU"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Payout",
                            (0, 1001000, 100000),
                            'combined_newcomb_avgq_eu_payout.pdf')

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [learned_actions_avgq, learned_actions_eu],
                            ["AVGQ", "EU"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Learned Action",
                            (0, 1.1, 0.2),
                            'combined_newcomb_avgq_eu_learned_action.pdf')