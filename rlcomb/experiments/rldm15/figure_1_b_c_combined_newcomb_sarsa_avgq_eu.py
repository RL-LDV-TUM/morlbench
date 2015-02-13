'''
Created on Feb 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Plot results from SARSA, AVGQ and EU Newcomb agent experiments.
'''

import sys
import os
sys.path.append(os.path.join('..', '..'))
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

    if not os.path.isfile("nc_sarsa_predicion_sweep.pickle") or \
            not os.path.isfile("nc_eu_prediction_sweep.pickle") or \
            not os.path.isfile("nc_avgq_prediction_sweep.pickle"):
        print >>sys.stderr, "run nc_sarsa_predicion_sweep.py, nc_eu_prediction_sweep\
            .py and nc_avgq_prediction_sweep.py first to create the \
            .pickle files"
        sys.exit(1)

    sarsaresults = None
    euresults = None
    avgqresults = None
    with open("nc_sarsa_predicion_sweep.pickle", 'rb') as f:
        sarsaresults = pickle.load(f)

    with open("nc_eu_prediction_sweep.pickle", 'rb') as f:
        euresults = pickle.load(f)

    with open("nc_avgq_prediction_sweep.pickle", 'rb') as f:
        avgqresults = pickle.load(f)

    avg_payout_sarsa = np.zeros((len(sarsaresults), linspace_steps))
    learned_actions_sarsa = np.zeros((len(sarsaresults), linspace_steps))
    avg_payout_eu = np.zeros((len(euresults), linspace_steps))
    learned_actions_eu = np.zeros((len(euresults), linspace_steps))
    avg_payout_avgq = np.zeros((len(avgqresults), linspace_steps))
    learned_actions_avgq = np.zeros((len(avgqresults), linspace_steps))

    for r in xrange(len(sarsaresults)):
        avg_payout_sarsa[r, :] = sarsaresults[r][0]
        learned_actions_sarsa[r, :] = sarsaresults[r][1]
    for r in xrange(len(euresults)):
        avg_payout_eu[r, :] = euresults[r][0]
        learned_actions_eu[r, :] = euresults[r][1]
    for r in xrange(len(avgqresults)):
        avg_payout_avgq[r, :] = avgqresults[r][0]
        learned_actions_avgq[r, :] = avgqresults[r][1]

    avg_payout_sarsa = avg_payout_sarsa.mean(axis=0)
    learned_actions_sarsa = learned_actions_sarsa.mean(axis=0)
    avg_payout_eu = avg_payout_eu.mean(axis=0)
    learned_actions_eu = learned_actions_eu.mean(axis=0)
    avg_payout_avgq = avg_payout_avgq.mean(axis=0)
    learned_actions_avgq = learned_actions_avgq.mean(axis=0)

    y_range = (300000, 1001000, 100000)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payout_sarsa, avg_payout_avgq, avg_payout_eu],
                            ["SARSA", "AVGQ", "EU"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'figure_1_c_combined_newcomb_sarsa_avg_eu_payout.pdf',
                            custom_yticks=["%iK" % (int(x/1000.0)) for x in
                                           np.arange(*y_range)],
                            fontsize=25,
                            label_fontsize=25,
                            label_offsets=[-30000, 0.0, 0])

    y_range = (0, 1.1, 0.2)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [learned_actions_sarsa, learned_actions_avgq,
                             learned_actions_eu],
                            ["SARSA", "AVGQ", "EU"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Learned Action",
                            y_range,
                            'figure_1_b_combined_newcomb_sarsa_avg_eu_learned_action.pdf',
                            custom_yticks=["1Box", "0.2\%", "0.4\%", "0.6\%", "0.8\%", "2Box"],
                            fontsize=25,
                            label_fontsize=25,
                            label_offsets=[0.2, 0.1, 0.0])