'''
Created on Feb, 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Plot results from SARSA, AVGQ Prisoner's Dilemma agent experiments.
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

    inputs = [("ppd_sarsa_local_prediction_sweep_normal.pickle", 'meta'),
              ("ppd_sarsa_total_prediction_sweep_normal.pickle", 'meta'),
              ("ppd_sarsa_local_prediction_sweep_modified.pickle", 'meta'),
              ("ppd_sarsa_total_prediction_sweep_modified.pickle", 'meta'),
              ("ppd_avgq_local_prediction_sweep_normal.pickle", 'meta'),
              ("ppd_avgq_total_prediction_sweep_normal.pickle", 'meta'),
              ("ppd_avgq_local_prediction_sweep_modified.pickle", 'meta'),
              ("ppd_avgq_total_prediction_sweep_modified.pickle", 'meta')]

    # TODO: make this dependent on inputs
    if reduce(lambda a, b: a or b, map(lambda n: not os.path.isfile(n[0]),
                                       inputs)):
        print >>sys.stderr, "run ppd_avgq_prediction_sweep.py, ppd_sarsa_prediction_sweep\
            .py first to create the .pickle files"
        sys.exit(1)

    ress = []
    for inp, meta in inputs:
        with open(inp, 'rb') as f:
            result = pickle.load(f)
            ress.append((result, meta))

    avg_payouts = []
    learned_actions = []

    for result, meta in ress:
        avg_payout = np.zeros((len(result), linspace_steps))
        learned_action = np.zeros((len(result), linspace_steps))
        for r in xrange(len(result)):
            avg_payout[r, :] = result[r][0]
            learned_action[r, :] = result[r][1]
        avg_payout = avg_payout.mean(axis=0)
        learned_action = learned_action.mean(axis=0)
        avg_payouts.append(avg_payout)
        learned_actions.append(learned_action)

    # TODO: this indexing is awkward, man why do you introduce 'metadata'
    #       above if you don't use it.
    y_range = (0, 7, 1)
    y_range_print = (1, 7, 1)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts[0], avg_payouts[1], avg_payouts[4],
                             avg_payouts[5]],
                            ["SARSA (I)", "SARSA (T)", "AVGQ (I)",
                             "AVGQ (T)"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'figure_2_c_combined_pd_sarsa_avg_payout_normal.pdf',
                            custom_yticks=[""] + ["%i" % (int(x)) for x in
                                                  np.arange(*y_range_print)],
                            fontsize=25,
                            label_fontsize=25,
                            y_lim=(0, 6),
                            label_offsets=[-0.1, -0.4, 0.0, 0.0])

    y_range = (0, 1100000, 100000)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts[2], avg_payouts[3], avg_payouts[6],
                             avg_payouts[7]],
                            ["SARSA (I)", "SARSA (T)", "AVGQ (I)",
                             "AVGQ (T)"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'figure_2_d_combined_pd_sarsa_avg_payout_modified.pdf',
                            custom_yticks=["%iK" % (int(x/1000.0)) for x in
                                           np.arange(*y_range)],
                            fontsize=25,
                            label_fontsize=25,
                            #y_lim=(0, 6),
                            label_offsets=[-60000, -1000, 0, 60000])