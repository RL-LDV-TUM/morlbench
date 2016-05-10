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


from morlbench.morl_agents import MORLChebyshevAgent, MORLHVBAgent, DynMultiCritAverageRewardAgent
from morlbench.morl_problems import MORLProblem, Deepsea, MORLGridworld
from morlbench.experiment_helpers import morl_interact_multiple
from morlbench.helpers import HyperVolumeCalculator


class TestAgents(unittest2.TestCase):

    def setUp(self):
        self.gridworldproblem = MORLGridworld()
        self.scalarization_weights = np.zeros(self.gridworldproblem.reward_dimension)
        self.scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)],len(self.scalarization_weights))
        self.tau = np.mean(self.scalarization_weights)
        self.ref = [-1.0, -1.0, -1.0]
        self.alf = 0.6
        self.eps = 0.6
        self.chebyagent = MORLChebyshevAgent(self.gridworldproblem, [1000, 1000, 1000], alpha=self.alf, epsilon=self.eps, tau=self.tau, ref_point=self.ref)
        self.hvbagent = MORLHVBAgent(self.gridworldproblem, alpha=self.alf, epsilon=self.eps, ref=self.ref, scal_weights=[1.0, 10.0])
        self.interactions = 500


class TestLearning(TestAgents):

    def runTest(self):
        self.runInteractions()
        self.runSelection()
        self.show_stats()

    def runSelection(self):
        new_state = self.chebyagent.decide(0, 3)
        print 'TEST(cheby): decision-from state 3-action:'+str(new_state)

        new_state2 = self.hvbagent.decide(0, 3)
        print 'TEST(hvb): decision-from state 3-action:'+str(new_state2)

    def runInteractions(self):
        payouts, moves, states = morl_interact_multiple(self.chebyagent, self.gridworldproblem, self.interactions,
                                                        max_episode_length=150)
        print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
              str(states[:]) + '\n')

        payouts2, moves2, states2 = morl_interact_multiple(self.hvbagent, self.gridworldproblem, self.interactions,
                                                        max_episode_length=150)
        print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
              str(states2[:]) + '\n')

    def show_stats(self):
        #plt.figure(0)
        a_list = self.chebyagent.max_volumes
        solution = len(a_list)/self.interactions
        # solution = 1
        u1 = [0]
        if len(a_list) % solution:
            for i in range(len(a_list) % solution):
                del a_list[len(a_list)-1]
        z = 0
        while z < len(a_list):
            u1.append(np.mean(a_list[z:z+solution]))
            z += solution
        x = np.arange(((len(a_list)/solution)-len(a_list) % solution)+1)
        v_list = self.hvbagent.max_volumes
        solution = len(v_list)/self.interactions
        u2 = [0]
        if len(v_list) % solution:
            for i in range(len(v_list) % solution):
                del v_list[len(v_list)-1]
        z = 0
        while z < len(v_list):
            u2.append(np.mean(v_list[z:z+solution]))
            z += solution
        del u2[len(u1):]
        # plt.subplot(211)
        x1, y1 = u1.index(max(u1)), max(u1)
        x2, y2 = u2.index(max(u2)), max(u2)
        plt.plot(x, u1, 'r', x, u2, 'b')
        plt.axis([0-0.01*len(u1), len(u1), 0, 1.1*max([max(u1), max(u2)])])
        plt.xlabel('step')
        plt.ylabel('hypervolume')
        plt.show()
        '''
        x = np.arange(((len(v_list)/solution)-len(v_list) % solution)+1)
        plt.subplot(212)
        plt.plot(x, u, 'r', label="hvb")
        plt.axis([0-0.1*len(u), len(u), 0, 1.1*max(u)])
        plt.xlabel('step')
        plt.ylabel('hypervolume')
        plt.show()
        '''

class TestHyperVolumeCalculator(unittest2.TestCase):
    def setUp(self):
        # create refpoints
        self.ref_point2d = [0.1, 0.1]
        self.ref_point3d = [0.1, 0.1, 0.1]
        # data set / random points between 0/0 - 1/1
        self.set2d = np.zeros((20, 2))
        self.set3d = np.zeros((100, 3))
        for i in range(20):
            for u in range(2):
                self.set2d[i, u] = random.random()
        for i in range(100):
            for u in range(3):
                self.set3d[i, u] = random.random()
        # initialize calculator
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
        # hv3d = self.hv_3d_calc.compute_hv(self.set3d)
        print hv

'''
class TestDynMultiCritAverageRewardAgent(unittest2.TestCase):
    def setUp(self):
        self.agent = DynMultiCritAverageRewardAgent()
        '''