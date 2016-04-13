'''
Created on Nov 22, 2012

@author: Dominik Meyer <meyerd@mytum.de>
'''

import sys
import os
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))

#import numpy as np
#import problems
#import agents

import logging as log

log.basicConfig(level=log.INFO)

from run_experiment import RunNewcombExperiment

if __name__ == '__main__':
    if len(sys.argv) < 2:
        log.error('usage: %s <experiment name> <plot filename>' % \
            (sys.argv[0]))
        sys.exit(1)

    experiment_name = sys.argv[1]
    plot_filename = sys.argv[2]
    if os.path.isfile(plot_filename):
        log.error('"%s" already exists' % (plot_filename))
        sys.exit(1)

    experiment = RunNewcombExperiment()

#    found_experiment_dir = experiment.get_exp(experiment_name)
    params = experiment.get_params(experiment_name)
    print params
    found_experiment_dir = None
    if os.path.isdir(params['path']):
        found_experiment_dir = params['path']

    if found_experiment_dir:
        log.info('found experiment dir in "%s"' % (experiment_name,
                                                   found_experiment_dir))
    else:
        log.error('coudn\'t find path for experiment "%s"' % (experiment_name))
