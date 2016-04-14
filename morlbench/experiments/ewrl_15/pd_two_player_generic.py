#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Apr 23, 2015

@author: Dominik Meyer <meyerd@mytum.de>
"""

"""
Let two prisoners dilemma players play agains each other.
"""

import sys
import os
sys.path.append(os.path.join('..', '..'))
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from experiment_helpers import interact_multiple_twoplayer
import problems
import agents


def onerun(r, aparams1, aparams2, pparams, expparams):
    lexpparams = expparams.copy()

    interactions = lexpparams['interactions']
    paramspace = lexpparams['paramspace']

    avg_payouts_in_run1 = []
    std_payouts_in_run1 = []
    avg_payouts_in_run2 = []
    std_payouts_in_run2 = []
    learned_actions_in_run1 = []
    learned_actions_in_run2 = []
    total_payouts_in_run = []

    for paramspace_val in paramspace:
        laparams1 = aparams1.copy()
        laparams2 = aparams2.copy()
        lpparams = pparams.copy()

        pclass = lpparams['_problem_class']
        lpparams.pop('_problem_class', None)
        problem = eval('problems.' + pclass)(**lpparams)
        if not expparams['paramspace_to_problem_parameter'] is None:
            lpparams[expparams['paramspace_to_problem_parameter']] = \
                paramspace_val
        a1class = laparams1['_agent_class']
        laparams1.pop('_agent_class')
        if not expparams['paramspace_to_agent1_parameter'] is None:
            laparams1[expparams['paramspace_to_agent1_parameter']] = \
                paramspace_val
        agent1 = eval('agents.' + a1class)(problem, **laparams1)
        a2class = laparams2['_agent_class']
        laparams2.pop('_agent_class')
        if not expparams['paramspace_to_agent2_parameter'] is None:
            laparams2[expparams['paramspace_to_agent2_parameter']] = \
                paramspace_val
        agent2 = eval('agents.' + a2class)(problem, **laparams2)

        log.info('Playing ...')
        log.info('%s' % (str(agent1)))
        log.info('%s' % (str(problem)))
        log.info(' VERSUS')
        log.info('%s' % (str(agent2)))
        log.info('%s' % (str(problem)))

        (_, payouts1), (_, payouts2) = \
            interact_multiple_twoplayer(agent1, agent2,
                                        problem, interactions,
                                        use_sum_of_payouts=lexpparams['use_sum_of_payouts'])
        avg_payout1 = payouts1.mean(axis=0)
        std_payout1 = payouts1.std(axis=0)
        avg_payout2 = payouts2.mean(axis=0)
        std_payout2 = payouts2.std(axis=0)
        total = payouts1 + payouts2
        avg_total = total.mean(axis=0)

        avg_payouts_in_run1.append(avg_payout1)
        std_payouts_in_run1.append(std_payout1)
        avg_payouts_in_run2.append(avg_payout2)
        std_payouts_in_run2.append(std_payout2)
        total_payouts_in_run.append(avg_total)

        learned_actions_in_run1.append(agent1.get_learned_action())
        learned_actions_in_run2.append(agent2.get_learned_action())

        log.info('Average Payout: %.3f vs. %.3f (total: %.3f)' %
                 (avg_payout1, avg_payout2, avg_total))

    return ((np.array(avg_payouts_in_run1),
             np.array(learned_actions_in_run1)),
            (np.array(avg_payouts_in_run2),
             np.array(learned_actions_in_run2)))


if __name__ == '__main__':
    def usage(name):
        print >>sys.stderr, "usage: %s [--only-update-params]" % (name)
        sys.exit(1)

    only_params = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "--only-update-params":
            only_params = True
        else:
            usage(sys.argv[0])

    from experiment_definitions import experiments

    for e in experiments:
        aparams1 = e['aparams1']
        aparams2 = e['aparams2']
        pparams = e['pparams']
        expparams = e['expparams']
        plotparams = e['plotparams']
        picklefile = e['picklefile']
        results = None

        results = None
        if only_params:
            with open(picklefile, 'rb') as f:
                resultstruct = pickle.load(f)
                results = resultstruct['results']
        else:
            if os.path.exists(picklefile):
                continue

            results = Parallel(n_jobs=-1)(delayed(onerun)(r, aparams1, aparams2,
                                                          pparams,
                                                          expparams) for r in
                                          xrange(expparams['independent_runs']))

        resultstruct = {
            'results': results,
            'aparams1': aparams1,
            'aparams2': aparams2,
            'pparams': pparams,
            'expparams': expparams,
            'plotparams': plotparams}
        with open(picklefile, 'wb') as f:
            pickle.dump(resultstruct, f)
