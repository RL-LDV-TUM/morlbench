'''
Created on Apr 23, 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

import numpy as np
import logging as log


def interact_multiple(agent, problem, interactions):
    '''
    Interact multiple times with the problem and then
    return arrays of actions chosen and payouts received
    in each stage.
    '''
    payouts = []
    actions = []
    log.info('Playing %i interactions ... ' % (interactions))
    for t in interactions:
        payouts = agent.interact(t)
        action = agent.decide(t)
        payout = problem.play(action)
        agent.learn(t, action, payout)
        log.debug(' step %05i: action: %i, payout: %i' %
                  (t, action, payout))
        payouts.append(payout)
        actions.append(action)
    payouts = np.array(payouts)
    actions = np.array(actions)
    return actions, payouts
