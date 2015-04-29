'''
Created on March, 27 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Plot results from SARSA vs. SARSA in PrisonersDilemma
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
    linspace_from = 0.5
    linspace_to = 0.99
    linspace_steps = 10

    inputs = [("pd_sarsa_local_prediction_sweep_normal.pickle", 'meta'),
              ("pd_sarsa_local_prediction_sweep_modified.pickle", 'meta')]

    if reduce(lambda a, b: a or b, map(lambda n: not os.path.isfile(n[0]),
                                       inputs)):
        print >>sys.stderr, "run pd_two_player_sarsa_vs_sarsa.py first to \
            create the .pickle files"
        sys.exit(1)

    ress = []
    for inp, meta in inputs:
        with open(inp, 'rb') as f:
            result = pickle.load(f)
            ress.append((result, meta))

    avg_payouts1 = []
    learned_actions1 = []
    avg_payouts2 = []
    learned_actions2 = []

    for result, meta in ress:
        avg_payout1 = np.zeros((len(result), linspace_steps))
        learned_action1 = np.zeros((len(result), linspace_steps))
        avg_payout2 = np.zeros((len(result), linspace_steps))
        learned_action2 = np.zeros((len(result), linspace_steps))
        for r in xrange(len(result)):
            avg_payout1[r, :] = result[r][0][0]
            avg_payout2[r, :] = result[r][1][0]
            learned_action1[r, :] = result[r][0][1]
            learned_action2[r, :] = result[r][1][1]
        avg_payout1 = avg_payout1.mean(axis=0)
        avg_payout2 = avg_payout2.mean(axis=0)
        learned_action1 = learned_action1.mean(axis=0)
        learned_action2 = learned_action2.mean(axis=0)
        avg_payouts1.append(avg_payout1)
        avg_payouts2.append(avg_payout2)
        learned_actions1.append(learned_action1)
        learned_actions2.append(learned_action2)

    # TODO: this indexing is awkward, man why do you introduce 'metadata'
    #       above if you don't use it.
    y_range = (0, 7, 1)
    y_range_print = (1, 7, 1)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts1[0], avg_payouts2[0]],
                            ["SARSA 1", "SARSA 2"],
                            r"$\epsilon$",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'pd_sarsa_avg_payout_normal.pdf',
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
                                         linspace_steps)],
                            [avg_payouts1[1], avg_payouts2[1]],
                            ["SARSA 1", "SARSA 2"],
                            r"$\epsilon$",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'pd_sarsa_avg_payout_modified.pdf',
                            custom_yticks=["%iK" % (int(x/1000.0)) for x in
                                           np.arange(*y_range)],
                            fontsize=25,
                            label_fontsize=25,
                            # y_lim=(0, 6),
                            label_offsets=[-60000, -1000, 0, 60000])

    # x_range = (0, 1.1, 0.2)
    x_range = (linspace_from - 0.1, linspace_to + 0.1, 0.2)
    y_range = (0, 1.1, 0.2)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [learned_actions1[0], learned_actions2[0],
                             learned_actions1[1], learned_actions2[1]],
                            ["SARSA 1 (I)", "SARSA 2 (I)",
                             "SARSA 1 (T)", "SARSA 2 (T)"],
                            r"$\epsilon$",
                            x_range,
                            "Learned Action",
                            y_range,
                            'pd_sarsa_learned_action.pdf',
                            custom_yticks=["Cooperate", "0.2\%", "0.4\%",
                                           "0.6\%", "0.8\%", "Defect"],
                            fontsize=25,
                            label_fontsize=25,
                            label_offsets=[0.2, 0.1, 0.0, -0.1])
