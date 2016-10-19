#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 19, 2012

@author: Dominik Meyer <meyerd@mytum.de>
@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
@author: Simon Woelzmueller   <ga35voz@mytum.de>

    Copyright (C) 2016  Dominik Meyer, Johannes Feldmaier, Simon Woelzmueller

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging as log
import numpy as np

log.basicConfig(level=log.DEBUG)

from morlbench.problems import Newcomb
from morlbench.agents import OneBoxNewcombAgent, TwoBoxNewcombAgent, SARSANewcombAgent
from morlbench.experiment_helpers import interact_multiple


if __name__ == '__main__':
    problem = Newcomb(predictor_accuracy=0.99,
                      payouts=np.array([[1000000, 0], [1001000, 1000]]))
    agent = SARSANewcombAgent(problem, alpha=0.1, gamma=0.9, epsilon=0.9)

    interactions = 10000

    log.info('Playing ...')
    log.info('%s' % (str(agent)))
    log.info('%s' % (str(problem)))

    _, payouts = interact_multiple(agent, problem, interactions)

    log.info('Average Payout: %f' % (payouts.mean(axis=0)))
