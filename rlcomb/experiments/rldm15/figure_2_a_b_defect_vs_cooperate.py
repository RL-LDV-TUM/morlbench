'''
Created on Feb, 2015

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

from problems import ProbabilisticPrisonersDilemma
from agents import DefectProbabilisticPrisonerAgent, \
                        CooperateProbabilisticPrisonerAgent
from experiment_helpers import interact_multiple


if __name__ == '__main__':
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    avg_payouts1 = []
    std_payouts1 = []
    avg_payouts2 = []
    std_payouts2 = []
    avg_total1 = []
    avg_total2 = []

    for cooperation_prob in np.linspace(linspace_from, linspace_to,
                                        linspace_steps):
        problem1 = ProbabilisticPrisonersDilemma(coop_p=cooperation_prob)
        problem2 = ProbabilisticPrisonersDilemma(coop_p=cooperation_prob)
#         problem1 = ProbabilisticPrisonersDilemma(T=1001000.0, R=50000.0,
#                                                     P=1000.0, S=0.0, coop_p=cooperation_prob)
#         problem2 = ProbabilisticPrisonersDilemma(T=1001000.0, R=50000.0,
#                                                     P=1000.0, S=0.0, coop_p=cooperation_prob)
        agent1 = DefectProbabilisticPrisonerAgent(problem1)
        agent2 = CooperateProbabilisticPrisonerAgent(problem2)

        log.info('Playing ...')
        log.info('%s' % (str(agent1)))
        log.info('%s' % (str(problem1)))
        log.info(' VERSUS')
        log.info('%s' % (str(agent2)))
        log.info('%s' % (str(problem2)))

        _, payouts1 = interact_multiple(agent1, problem1, interactions)
        total1 = payouts1.sum(axis=1)
        payouts1 = payouts1[:, 0]
        _, payouts2 = interact_multiple(agent2, problem2, interactions)
        total2 = payouts2.sum(axis=1)
        payouts2 = payouts2[:, 0]

        avg_payout1 = payouts1.mean(axis=0)
        std_payout1 = payouts1.std(axis=0)
        avg_payout2 = payouts2.mean(axis=0)
        std_payout2 = payouts2.std(axis=0)
        avg_totall1 = total1.mean(axis=0)
        avg_totall2 = total2.mean(axis=0)

        avg_payouts1.append(avg_payout1)
        std_payouts1.append(std_payout1)
        avg_payouts2.append(avg_payout2)
        std_payouts2.append(std_payout2)
        avg_total1.append(avg_totall1)
        avg_total2.append(avg_totall2)

        log.info('Average Payout: %.3f vs. %.3f (total: %.3f vs. %.3f)' %
                 (avg_payout1, avg_payout2, avg_totall1, avg_totall2))

    avg_payouts1 = np.array(avg_payouts1)
    std_payouts1 = np.array(std_payouts1)
    avg_payouts2 = np.array(avg_payouts2)
    std_payouts2 = np.array(std_payouts2)
    avg_total1 = np.array(avg_total1)
    avg_total2 = np.array(avg_total2)

    y_range = (0, 7, 1)
    y_range_print = (1, 7, 1)
    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts1, avg_payouts2],
                            ["Defect", "Cooperate"],
                            "Cooperation Probability",
                            (0, 1.1, 0.2),
                            "Payout",
                            y_range,
                            'figure_2_a_defect_vs_cooperate_payout.pdf',
                            custom_yticks=[""] + ["%i" % (int(x)) for x in
                                                  np.arange(*y_range_print)],
                            fontsize=25,
                            label_fontsize=25,
                            y_lim=(0, 6))

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_total1, avg_total2],
                            ["Defect", "Cooperate"],
                            "Cooperation Probability",
                            (0, 1.1, 0.2),
                            "Payout",
                            (0, 7, 1),
                            'figure_2_b_defect_vs_cooperate_total_payout.pdf',
                            custom_yticks=[""] + ["%i" % (int(x)) for x in
                                                  np.arange(*y_range_print)],
                            fontsize=25,
                            label_fontsize=25,
                            y_lim=(0, 6))