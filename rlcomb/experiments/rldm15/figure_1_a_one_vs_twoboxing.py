'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

import sys
import os
sys.path.append(os.path.join('..', '..'))
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np

log.basicConfig(level=log.DEBUG)

from plotting_stuff import plot_that_pretty_rldm15

from problems import Newcomb, RandomNewcomb
from agents import OneBoxNewcombAgent, TwoBoxNewcombAgent


if __name__ == '__main__':
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    avg_payouts1 = []
    avg_payouts2 = []

    for predictor_accuracy in np.linspace(linspace_from, linspace_to, 
                                          linspace_steps):
        problem1 = RandomNewcomb(predictor_accuracy=predictor_accuracy,
                          payouts=np.array([[1000000, 0], [1001000, 1000]]))
        problem2 = RandomNewcomb(predictor_accuracy=predictor_accuracy,
                          payouts=np.array([[1000000, 0], [1001000, 1000]]))
        agent1 = OneBoxNewcombAgent(problem1)
        agent2 = TwoBoxNewcombAgent(problem2)

        log.info('Playing ...')
        log.info('%s' % (str(agent1)))
        log.info('%s' % (str(problem1)))
        log.info(' VERSUS')
        log.info('%s' % (str(agent2)))
        log.info('%s' % (str(problem2)))

        payouts1 = agent1.interact_multiple(interactions)
        payouts2 = agent2.interact_multiple(interactions)
        avg_payout1 = payouts1.mean(axis=0)
        avg_payout2 = payouts2.mean(axis=0)

        avg_payouts1.append(avg_payout1)
        avg_payouts2.append(avg_payout2)

        log.info('Average Payout: %.3f vs. %.3f' % (avg_payout1, avg_payout2))

    avg_payouts1 = np.array(avg_payouts1)
    avg_payouts2 = np.array(avg_payouts2)

    y_range = (0, 1001000, 100000)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts1, avg_payouts2],
                            ["TwoBoxer", "OneBoxer"],
                            "Prediction Accuracy",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'figure_1_a_one_vs_twoboxing.pdf',
                            custom_yticks=["%iK" % (int(x/1000.0)) for x in
                                           np.arange(*y_range)],
                            fontsize=25,
                            label_fontsize=25)
