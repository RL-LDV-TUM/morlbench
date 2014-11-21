'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

import sys
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import matplotlib.pyplot as plt

log.basicConfig(level=log.DEBUG)

from problems import Newcomb, RandomNewcomb
from agents import OneBoxNewcombAgent, TwoBoxNewcombAgent


if __name__ == '__main__':
    interactions = 1000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 1000

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

        payouts1 = agent1.interact(interactions)
        payouts2 = agent2.interact(interactions)
        avg_payout1 = payouts1.mean(axis=0)
        avg_payout2 = payouts2.mean(axis=0)

        avg_payouts1.append(avg_payout1)
        avg_payouts2.append(avg_payout2)

        log.info('Average Payout: %.3f vs. %.3f' % (avg_payout1, avg_payout2))

    avg_payouts1 = np.array(avg_payouts1)
    avg_payouts2 = np.array(avg_payouts2)

    fig = plt.figure()
    plt.xlabel('prediction accuracy')
    plt.ylabel('payout')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), avg_payouts1, label='OneBoxer')
    plt.plot(np.linspace(linspace_from, linspace_to,
                         linspace_steps), avg_payouts2, label='TwoBoxer')
    plt.legend(loc='center right')
    plt.savefig("one_vs_two_box.png")
