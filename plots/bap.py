#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mar 03, 2016

@author: Johannes Feldmaier <@tum.de>
"""

import logging as log
import numpy as np
import sys

import cPickle as pickle

#log.basicConfig(level=log.DEBUG)
log.basicConfig(level=log.INFO)


from morlbench.morl_problems import MORLBuridansAssProblem, MORLBuridansAss1DProblem
from morlbench.morl_agents import MORLScalarizingAgent, QMorlAgent, PreScalarizedQMorlAgent, SARSALambdaMorlAgent,\
    SARSAMorlAgent, MORLHVBAgent
from morlbench.morl_policies import PolicyFromAgent, PolicyGridworld
from morlbench.inverse_morl import InverseMORLIRL
from morlbench.plot_heatmap import policy_plot2, transition_map, heatmap_matplot, policy_heat_plot
from morlbench.dynamic_programming import MORLDynamicProgrammingPolicyEvaluation, MORLDynamicProgrammingInverse
from morlbench.experiment_helpers import morl_interact_multiple_episodic, morl_interact_multiple_average_episodic
from morlbench.plotting_stuff import plot_hypervolume

import pickle
import random
import time

import matplotlib.pyplot as plt


if __name__ == '__main__':
    random.seed(218)
    np.random.seed(3)
    saved_weights = []
    plt.ion()
    problem = MORLBuridansAss1DProblem()
    scalarization_weights = np.array([1.0, 0.0, 0.0])
    eps = 0.9
    alfa = 0.08
    tau = 2.0
    ref_point = [-1.0, ]*problem.reward_dimension
    interactions = 1500
    chebyagent = MORLScalarizingAgent(problem, epsilon=eps, alpha=alfa, scalarization_weights=scalarization_weights,
                                      ref_point=ref_point, tau=tau, gamma=0.9, function='linear')

    payouts, moves, states = morl_interact_multiple_episodic(chebyagent, problem, interactions=interactions,
                                                             max_episode_length=300)
    log.info('Average Payout: %s' % (str(payouts.mean(axis=0))))

    volumes = [0]
    volumes.extend(chebyagent.max_volumes)
    x = np.arange(len(volumes))

    ##################################
    #               PLOT             #
    ##################################
    plt.figure()
    for i in volumes:
        plt.plot(x, volumes, 'b')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.axis([0 - 0.01 * len(x), len(x), 0, 1.1 * max(volumes)])
    plt.xlabel('interactions')
    plt.ylabel('hypervolume')
    plt.grid(True)
    plt.savefig('bap.pdf')
