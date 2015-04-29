'''
Created on March, 27 2015

@author: Dominik Meyer <meyerd@mytum.de>
'''

'''
Plot results from Agent vs. Agent in Prisoners Dilemma
'''

import sys
import os
sys.path.append(os.path.join('..', '..'))
sys.path.append('..')
sys.path.append('.')
import logging as log
import numpy as np
import os
import cPickle as pickle

log.basicConfig(level=log.INFO)

from plotting_stuff import plot_that_pretty_rldm15


if __name__ == '__main__':
    from experiment_definitions import experiments

    inputs = [e['picklefile'] for e in experiments]

    if reduce(lambda a, b: a or b, map(lambda n: not os.path.isfile(n),
                                       inputs)):
        print >>sys.stderr, "run pd_two_player_generic.py first to \
            create the .pickle files"
        sys.exit(1)

    for e in experiments:
        with open(e['picklefile']) as f:
            resultstruct = pickle.load(f)
            results = resultstruct['results']
            aparams1 = resultstruct['aparams1']
            aparams2 = resultstruct['aparams2']
            pparams = resultstruct['pparams']
            expparams = resultstruct['expparams']

            plotparams = resultstruct['plotparams']

            for p in plotparams:
                p['xdata'] = p['xdata'](results, e)
                p['ydata'] = p['ydata'](results, e)
                plot_that_pretty_rldm15(**p)
