#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 21, 2012

@author: Dominik Meyer <meyerd@mytum.de>
"""

"""
Experiment that sweeps over the prediction accuracy of the
Newcomb problems predictor and does a UCB1 learning over 
10000 iterations.
"""

import sys
import os
sys.path.append(os.path.join('..', '..'))
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

from joblib import Parallel, delayed

log.basicConfig(level=log.INFO)

from plotting_stuff import plot_that_pretty_rldm15

from problems import Newcomb
from agents import EUNewcombAgent
from experiment_helpers import interact_multiple


if __name__ == '__main__':
    independent_runs = 50
    interactions = 10000

    linspace_from = 0.01
    linspace_to = 0.99
    linspace_steps = 100

    loadresults = False

    avg_payouts = np.zeros((independent_runs, linspace_steps))
    learned_actions = np.zeros((independent_runs, linspace_steps))

    def onerun(r):
        avg_payouts_in_run = []
        learned_actions_in_run = []

        for prediction_accuracy in np.linspace(linspace_from, linspace_to,
                                               linspace_steps):
            problem = Newcomb(predictor_accuracy=prediction_accuracy,
                              payouts=np.array([[1000000, 0],
                                                [1001000, 1000]]))
            agent = EUNewcombAgent(problem)

            log.info('Playing ...')
            log.info('%s' % (str(agent)))
            log.info('%s' % (str(problem)))

            _, payouts = interact_multiple(agent, problem, interactions)
            avg_payout = payouts.mean(axis=0)
            avg_payouts_in_run.append(avg_payout)

            log.info('Average Payout for predicion accuraccy %.3f: %.3f' % \
                     (prediction_accuracy, avg_payout))

            learned_actions_in_run.append(agent.get_learned_action())

        return (np.array(avg_payouts_in_run), np.array(learned_actions_in_run))

    results = None

    if loadresults:
        with open("nc_eu_prediction_sweep.pickle", 'rb') as f:
            results = pickle.load(f)
    else:
        results = Parallel(n_jobs=-1)(delayed(onerun)(r) for r in
                                     xrange(independent_runs))
        with open("nc_eu_prediction_sweep.pickle", 'wb') as f:
            pickle.dump(results, f)