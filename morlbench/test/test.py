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
from morlbench.morl_problems import MORLProblem, Deepsea, MORLGridworld, MORLBurdiansAssProblem
from morlbench.experiment_helpers import morl_interact_multiple
from morlbench.helpers import HyperVolumeCalculator


class TestAgents(unittest2.TestCase):

    def setUp(self):
        # create Problem
        self.gridworldproblem = MORLGridworld()
        self.buridansass = MORLBurdiansAssProblem()
        # create an initialize randomly a weight vector
        self.scalarization_weights = np.zeros(self.gridworldproblem.reward_dimension)
        self.scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)],
                                                   len(self.scalarization_weights))
        # tau is for chebyshev agent
        self.tau = 1.0
        # ref point is used for Hypervolume calculation
        self.ref = [-1.0, -1.0, -1.0]
        # learning rate
        self.alf = 0.1
        # Propability of epsilon greedy selection
        self.eps = 0.1
        # create one agent using chebyshev scalarization method
        self.chebyagent = MORLChebyshevAgent(self.buridansass, [1.0, 1.0, 1.0], alpha=self.alf, epsilon=self.eps,
                                             tau=self.tau, ref_point=self.ref)
        # create one agent using Hypervolume based Algorithm
        self.hvbagent = MORLHVBAgent(self.buridansass,alpha=self.alf, epsilon=self.eps, ref=self.ref,
                                     scal_weights=[1.0, 10.0])
        # both agents interact (times):
        self.interactions = 20


class TestLearning(TestAgents):

    def runTest(self):
        self.runInteractions()
        self.runSelection()
        self.show_stats()
        self.testWeightVariation()

    def runSelection(self):
        # make just one decision for each learned agent, to see if they act the same way
        new_state = self.chebyagent.decide(0, 3)
        print 'TEST(cheby): decision-from state 3-action:'+str(new_state)

        new_state2 = self.hvbagent.decide(0, 3)
        print 'TEST(hvb): decision-from state 3-action:'+str(new_state2)

    def runInteractions(self):
        # make the interactions
        payouts, moves, states = morl_interact_multiple(self.chebyagent, self.gridworldproblem, self.interactions,
                                                        max_episode_length=150)
        print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
              str(states[:]) + '\n')

        payouts2, moves2, states2 = morl_interact_multiple(self.hvbagent, self.gridworldproblem, self.interactions,
                                                           max_episode_length=150)
        print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
              str(states2[:]) + '\n')

    def show_stats(self):
        # extract all volumes of each agent
        a_list = self.chebyagent.max_volumes
        v_list = self.hvbagent.max_volumes
        # solution = len(a_list)/self.interactions
        solution = 1
        # data preparation for another solution !=1
        if solution != 1:
            # nice curve contains (0,0)
            u1 = [0]
            # cut the longer list
            overlay = len(v_list) % solution
            if overlay:
                del a_list[len(a_list)-overlay:]
            z = 0
            # append mean values
            while z < len(a_list):
                u1.append(np.mean(a_list[z:z+solution]))
                z += solution
            # create x vector
            x = np.arange(((len(a_list)/solution)-len(a_list) % solution)+1)
            # solution = len(v_list)/self.interactions
            u2 = [0]
            # cut the longer list
            overlay = len(v_list) % solution

            if overlay:
                del v_list[len(v_list)-overlay:]
            z = 0
            while z < len(v_list):
                u2.append(np.mean(v_list[z:z+solution]))
                z += solution
        else:
            # just create two lists containing (0,0)
            u1 = [0]
            u1.extend(a_list)
            u2 = [0]
            u2.extend(v_list)
            x = np.arange(min([len(u1), len(u2)]))
        # cut longer list
        if len(u2) > len(u1):
            del u2[len(u1):]
        else:
            del u1[len(u2):]

        ##################################
        #               PLOT             #
        ##################################

        x1, y1 = u1.index(max(u1)), max(u1)
        x2, y2 = u2.index(max(u2)), max(u2)
        paretofront = [max([max(u1), max(u2)]), ]*len(x)
        plt.plot(x, u1, 'r', label="Chebyshev-Agent")
        plt.plot(x, u2, 'b', label="HVB-Agent")
        plt.plot(x, paretofront, 'g--', label="Paretofront")
        plt.legend(loc='lower right', frameon=False)
        plt.axis([0-0.01*len(u1), len(u1), 0, 1.1*max([max(u1), max(u2)])])
        plt.xlabel('interactions')
        plt.ylabel('hypervolume')
        plt.grid(True)
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

    def testWeightVariation(self):
        """
        this test creates 6 different chebyshev agents whose weights are each different. in the end it compares hvs
        :return:
        """
        # list of agents
        self.agents = []
        # list of volumes
        self.vollist = []
        # 6 agents with each different weights
        self.agents.append(MORLChebyshevAgent(self.gridworldproblem, [1.0, 0.0, 0.0], alpha=self.alf, epsilon=self.eps,
                                              tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLChebyshevAgent(self.gridworldproblem, [0.0, 1.0, 0.0], alpha=self.alf, epsilon=self.eps,
                                              tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLChebyshevAgent(self.gridworldproblem, [0.5, 0.5, 0.0], alpha=self.alf, epsilon=self.eps,
                                              tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLChebyshevAgent(self.gridworldproblem, [0.0, 0.5, 0.5], alpha=self.alf, epsilon=self.eps,
                                              tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLChebyshevAgent(self.gridworldproblem, [0.5, 0.0, 0.5], alpha=self.alf, epsilon=self.eps,
                                              tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLChebyshevAgent(self.gridworldproblem, [0.33, 0.33, 0.33], alpha=self.alf,
                                             epsilon=self.eps, tau=self.tau, ref_point=self.ref))

        # interact with each
        for agent in self.agents:
            p, a, s = morl_interact_multiple(agent, self.gridworldproblem, self.interactions,
                                             max_episode_length=150)
            # store all volumes containing (0,0)
            maxvol = [0]
            maxvol.extend(agent.max_volumes)
            self.vollist.append(maxvol)

        # cut longer lists
        length = min([len(x) for x in self.vollist])
        for lists in self.vollist:
            del lists[length:]
        # create x vectors
        x = np.arange(length)
        # colour vector
        colours = ['r', 'b', 'g', 'k', 'y', 'm']
        for lists in self.vollist:
            # printed name for label
            weights = self.agents[self.vollist.index(lists)]._w
            name = 'weights:'
            for i in range(len(weights)):
                name += str(weights[i])+'_'
            # no last underline
            name = name[:len(name)-1]
            # plotting
            plt.plot(x, lists, colours[self.vollist.index(lists)], label=name)
        # size of axes
        plt.axis([0-0.01*len(x), len(x), 0, 1.1*max([max(x) for x in self.vollist])])
        # position the legend
        plt.legend(loc='lower right', frameon=False)
        # show!
        plt.show()


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
        self.runPareto()
        self.runCompute()
        pass

    def runPareto(self):
        pf = self.hv_2d_calc.extract_front(self.set2d)

        plt.axis([0-0.1, max(self.set2d[:, 0]*1.21), 0-0.1, max(self.set2d[:, 1]*1.1)])
        pfx = [pf[i][0] for i in range(len(pf))]
        pfy = [pf[u][1] for u in range(len(pf))]

        plt.plot(self.set2d[:, 0], self.set2d[:, 1], 'ro', pfx, pfy)
        plt.xlabel('1')
        plt.ylabel('2')
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

    def runCompute(self):
        hv = self.hv_2d_calc.compute_hv(self.set2d)
        hv3d = self.hv_3d_calc.compute_hv(self.set3d)
        print hv
        print hv3d


class TestProblems(unittest2.TestCase):
    def setUp(self):
        self.problem = MORLBurdiansAssProblem()


class TestBuridan(TestProblems):
    def runTest(self):
        self.testDistance()
        self.testReward()
        self.testPlay()
        self.testFoodstolen()

    def testDistance(self):
        # tests if the distance is calculated correctly
        i = 1, 1
        u = 2, 1
        dist = self.problem._get_distance(i, u)
        self.assertEqual(dist, 1.0, 'wrong distance')

    def testReward(self):
        # tests if the reward of staying in the same position gives zero reward
        self.problem.reset()
        reward = self.problem.play(0)
        reward = tuple([e for e in reward])
        self.assertTupleEqual(reward, (0, 0, 0), 'wrong reward')

    def testPlay(self):
        # tests if we get negative hunger reward if we go on 10 steps without eating
        self.problem.reset()
        order = [1, 2, 3, 4, 4, 3, 2, 2, 1]
        for i in order:
            r = self.problem.play(i)
        reward = tuple([e for e in r])
        self.assertTupleEqual(reward, (-1.0, 0.0, -1.0), 'reward of hunger doesn\'t fit')

    def testFoodstolen(self):
        # sets the stealing probability to 100% and tests if the food is stolen
        # before set the problem to init state (middle)
        self.problem.reset()
        temp = self.problem.steal_probability
        self.problem.steal_probability = 1
        # go upwards, beacause we get away from bottom right food visible neighborhood
        reward = self.problem.play(2)
        # reward for stealing is second dimension
        steal_rew = reward[1]
        self.assertEqual(steal_rew, -0.5, 'no stealing reward')

    def testPrint(self):
        self.problem.reset()
        # go down
        self.problem.play(4)
        # check on map if he's gone down
        self.problem.print_map(self.problem._get_position(self.problem.state))

    def testFoodEaten(self):
        self.problem.reset()
        # go upwards
        self.problem.play(2)
        # and then left
        self.problem.play(3)
        # and stay there, to get food reward
        reward = self.problem.play(0)[0]
        self.assertEqual(reward, 1.0, 'got wrong food reward')