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

        payouts1, total1 = agent1.interact_multiple(interactions,
                                                    total_payout=True)
        payouts2, total2 = agent2.interact_multiple(interactions,
                                                    total_payout=True)
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

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps),
                             np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_payouts1, avg_payouts2],
                            ["Defect", "Cooperate"],
                            "Cooperation Probability",
                            (0, 1.1, 0.2),
                            "Payout",
                            (0, 6, 1),
                            'figure_2_a_defect_vs_cooperate_payout.pdf')

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
                            'figure_2_b_defect_vs_cooperate_total_payout.pdf')