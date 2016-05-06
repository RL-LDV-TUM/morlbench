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
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from morlbench.morl_agents import MORLChebyshevAgent, MORLHVBAgent
from morlbench.morl_problems import MORLProblem, Deepsea, MORLGridworld
from morlbench.experiment_helpers import morl_interact_multiple
from morlbench.helpers import HyperVolumeCalculator


class TestMORLChebishevAgent(unittest2.TestCase):

    def setUp(self):
        self.problem = MORLGridworld()
        self.scalarization_weights = np.zeros(self.problem.reward_dimension)
        self.scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)],
                                                   len(self.scalarization_weights))
        self.tau = np.mean(self.scalarization_weights)
        self.agent = MORLChebyshevAgent(self.problem, self.scalarization_weights, alpha=0.3, epsilon=0.4, tau=self.tau)
        self.interactions = 100

'''
class TestLearning(TestMORLChebishevAgent):

    def runTest(self):
        # self.runInteractions()
        # self.runSelection()
        pass

    def runSelection(self):
        new_state = self.agent.decide(0, 3)
        print 'TEST: decision-from state 3-action:'+str(new_state)

    def runInteractions(self):
        payouts, moves, states = morl_interact_multiple(self.agent, self.problem, self.interactions,
                                                        max_episode_length=150)
        print("TEST: interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " + str(states[:]) + '\n')

'''


class TestMORLHVBAgent(unittest2.TestCase):

    def setUp(self):
        self.problem = Deepsea()
        self.agent = MORLHVBAgent(self.problem, alpha=0.5, epsilon=0.6, ref=[-25.0, -5.0])
        self.interactions = 50


class TestLearning(TestMORLHVBAgent):

    def runTest(self):

        self.runInteractions()
        self.runSelection()
        self.show_stats()

    def runSelection(self):
        new_state = self.agent.decide(0, 3)
        print 'TEST(HVB): decision-from state 3-action:'+str(new_state)

    def runInteractions(self):
        payouts, moves, states = morl_interact_multiple(self.agent, self.problem, self.interactions,
                                                        max_episode_length=150)
        print("TEST(HVB): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " + str(states[:]) + '\n')

    def show_stats(self):
        a_list = self.agent.max_volumes
        solution = 1
        u = []
        if len(a_list) % solution:
            for i in range(len(a_list) % solution):
                del a_list[len(a_list)-1]
        z = 0
        while z < len(a_list):
            u.append(np.mean(a_list[z:z+solution]))
            z += solution
        x = np.arange((len(a_list)/solution)-len(a_list) % solution)
        plt.plot(x, u, 'r')
        plt.axis([0-0.1*len(u), len(u), 0, 1.1*max(u)])
        plt.show()

'''
class TestHyperVolumeCalculator(unittest2.TestCase):
    def setUp(self):
        self.ref_point2d = [0.4, 0.4]
        self.ref_point3d = [0.1, 0.1, 0.1]
        self.set2d = np.zeros((20, 2))
        self.set3d = np.zeros((100, 3))
        for i in range(20):
            for u in range(2):
                self.set2d[i, u] = random.random()
        for i in range(100):
            for u in range(3):
                self.set3d[i, u] = random.random()

        self.hv_2d_calc = HyperVolumeCalculator(self.ref_point2d)
        self.hv_3d_calc = HyperVolumeCalculator(self.ref_point3d)


class TestCalculation(TestHyperVolumeCalculator):
    def runTest(self):
        # self.runPareto()
        # self.runCompute()
        pass

    def runPareto(self):
        pf = self.hv_2d_calc.extract_front(self.set2d)

        plt.axis([0-0.1, max(self.set2d[:, 0]*1.21), 0-0.1, max(self.set2d[:, 1]*1.1)])
        pfx = [pf[i][0] for i in range(len(pf))]
        pfy = [pf[u][1] for u in range(len(pf))]

        plt.plot(self.set2d[:, 0], self.set2d[:, 1], 'ro', pfx, pfy)
        plt.xlabel('1')
        plt.ylabel('2')
        # print(str(state_distribution.max())+str(state_distribution.argmax()))
        plt.grid(False)
        plt.show()

        pf3d = self.hv_3d_calc.extract_front(self.set3d)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        for i in range(len(self.set3d)):
            apx = self.set3d[i][0]
            apy = self.set3d[i][1]
            apz = self.set3d[i][2]
            ax.scatter(apx, apy, apz, 'b')
        for u in range(len(pf3d)):
            pf3dx = pf3d[u][0]
            pf3dy = pf3d[u][1]
            pf3dz = pf3d[u][2]
            ax.scatter(pf3dx, pf3dy, pf3dz, 'r', marker='^')
        plt.show()
        #print pf
        #print self.set3d
        #print pf3d

    def runCompute(self):
        hv = self.hv_2d_calc.compute_hv(self.set2d)
        #hv3d = self.hv_3d_calc.compute_hv(self.set3d)
        print hv
        #print hv3d'''