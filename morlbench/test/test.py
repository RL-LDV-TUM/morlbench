#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Apr 21, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>
"""

import unittest2
import numpy as np
import matplotlib.pyplot as plt
import random


from morlbench.morl_agents import MORLChebyshevAgent
from morlbench.morl_problems import MORLProblem, Deepsea
from morlbench.experiment_helpers import morl_interact_multiple
from morlbench.helpers import HyperVolumeCalculator


class TestMORLChebishevAgent(unittest2.TestCase):

    def setUp(self):
        self.problem = Deepsea(extended_reward=False)
        self.scalarization_weights = np.zeros(self.problem.reward_dimension)
        self.scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)],
                                                   len(self.scalarization_weights))
        self.tau = np.mean(self.scalarization_weights)
        self.agent = MORLChebyshevAgent(self.problem, self.scalarization_weights, alpha=0.3, epsilon=0.4, tau=self.tau)
        self.interactions = 4


class TestLearning(TestMORLChebishevAgent):

    def runTest(self):
        payouts, moves, states = morl_interact_multiple(self.agent, self.problem, self.interactions,
                                                        max_episode_length=150)
        print("P: "+str(payouts[:])+", M: " + str(moves[:]) + " S: " + str(states[:]) + '\n')


class TestHyperVolumeCalculator(unittest2.TestCase):
    def setUp(self):
        self.ref_point2d = [0.1, 0.1]
        self.ref_point3d = [0.1, 0.1, 0.1]
        self.set2d = np.random.rand(20, 2)
        self.set3d = np.random.rand(20, 3)
        self.hv_2d_calc = HyperVolumeCalculator(self.ref_point2d, self.set2d)
        self.hv_3d_calc = HyperVolumeCalculator(self.ref_point3d, self.set3d)

class TestCalculation(TestHyperVolumeCalculator):
    def runTest(self):
        self.runPareto()

    def runPareto(self):
        pf = self.hv_2d_calc.extract_front(self.set2d)

        plt.axis([0-0.1, max(self.set2d[:, 0]*1.21), 0-0.1, max(self.set2d[:, 1]*1.1)])
        pfx = [pf[i][0] for i in range(len(pf))]
        pfy = [pf[u][1] for u in range(len(pf))]
        plt.plot(self.set2d[:, 0], self.set2d[:, 1], 'ro', pfx, pfy, 'bs')
        plt.xlabel('1')
        plt.ylabel('2')
        # print(str(state_distribution.max())+str(state_distribution.argmax()))
        plt.grid(False)
        plt.show()

        pf3d =self.hv_3d_calc.extract_front(self.set3d)
        # TODO check pf3d function

        print pf
        print pf3d

