#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Apr 23, 2015

@author: Dominik Meyer <meyerd@mytum.de>
"""

import sys
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np

log.basicConfig(level=log.DEBUG)

from plotting_stuff import plot_that_pretty_rldm15

from problems import PrisonersDilemma
from agents import DefectPrisonerAgent, \
                        CooperatePrisonerAgent
from experiment_helpers import interact_multiple, interact_multiple_twoplayer


if __name__ == '__main__':
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 5

    avg_payouts1 = []
    std_payouts1 = []
    avg_payouts2 = []
    std_payouts2 = []
    avg_totals = []

    for cooperation_prob in np.linspace(linspace_from, linspace_to,
                                        linspace_steps):
#         problem = PrisonersDilemma(coop_p=cooperation_prob)
        problem = PrisonersDilemma()
        agent1 = DefectPrisonerAgent(problem)
        agent2 = CooperatePrisonerAgent(problem)

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

        avg_payouts1.append(avg_payout1)
        std_payouts1.append(std_payout1)
        avg_payouts2.append(avg_payout2)
        std_payouts2.append(std_payout2)
        avg_totals.append(avg_total)

        log.info('Average Payout: %.3f vs. %.3f (total: %.3f)' %
                 (avg_payout1, avg_payout2, avg_total))

    avg_payouts1 = np.array(avg_payouts1)
    std_payouts1 = np.array(std_payouts1)
    avg_payouts2 = np.array(avg_payouts2)
    std_payouts2 = np.array(std_payouts2)
    avg_totals = np.array(avg_totals)

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
                            'pd_defect_vs_cooperate.pdf')

    plot_that_pretty_rldm15([np.linspace(linspace_from, linspace_to,
                                         linspace_steps)],
                            [avg_totals],
                            ["Defect vs. Cooperate"],
                            "Cooperation Probability",
                            (0, 1.1, 0.2),
                            "Payout",
                            (0, 7, 1),
                            'pd_defect_vs_cooperate_total_payout.pdf')
