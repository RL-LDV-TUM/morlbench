'''
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

import logging as log
import numpy as np

log.basicConfig(level=log.DEBUG)

from problems import Newcomb
from agents import OneBoxNewcombAgent, TwoBoxNewcombAgent


if __name__ == '__main__':
    problem = Newcomb(predictor_accuracy=0.99,
                      payouts=np.array([[1000000, 0], [1001000, 1000]]))
    agent = TwoBoxNewcombAgent(problem)

    interactions = 100000

    log.info('Playing ...')
    log.info('%s' % (str(agent)))
    log.info('%s' % (str(problem)))

    payouts = agent.interact(interactions)

    log.info('Average Payout: %f' % (payouts.mean(axis=0)))
