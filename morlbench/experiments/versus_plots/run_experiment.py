'''
Created on Nov 22, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Load a experiment .cfg file and run it.
'''

import sys
import os
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))

import numpy as np
import problems
import agents

import logging as log

log.basicConfig(level=log.INFO)

from expsuite import PyExperimentSuite


class RunNewcombExperiment(PyExperimentSuite):
    restore_supported = False

    def reset(self, params, rep):
        self.problem = eval('problems.' + params['problem'] + '()')
        self.problem.predictor_accuracy = params['predictor_accuracy']
        self.problem.payouts = eval(params['payouts'])
        self.agent = eval('agents.' + params['agent'] + '(self.problem)')
        self.agent.alpha = params['alpha']
        self.agent.gamma = params['gamma']
        self.agent.epsilon = params['epsilon']
        log.debug('Playing ...')
        log.debug('%s' % (str(self.agent)))
        log.debug('%s' % (str(self.problem)))
        log.debug(' for %i iterations' % (params['iterations']))

    def iterate(self, params, rep, n):
        action = self.agent.decide(n)
        payout = self.problem.play(action)
        learned_action = self.agent.get_learned_action()

        log.debug('Learned action %i in iteration %i (payout %.3f)' %
                     (learned_action, n, payout))

        ret = {'iteration': n, 'rep': rep, 'action_executed': action,
               'payout': payout, 'learned_action': learned_action}
        return ret


if __name__ == '__main__':
    experiment = RunNewcombExperiment()
    experiment.start()
