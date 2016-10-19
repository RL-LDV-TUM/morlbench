#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Apr 21, 2016

@author: Simon Wölzmüller <ga35voz@mytum.de>

    Copyright (C) 2016  Simon Woelzmueller

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

import unittest2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.spatial import ConvexHull

from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent, MORLHLearningAgent
from morlbench.morl_agents_multiple_criteria import MORLConvexHullValueIteration
from morlbench.morl_problems import MORLGridworld, MORLBuridansAssProblem, MOPuddleworldProblem, \
    MORLResourceGatheringProblem, Deepsea, MountainCar, MORLBuridansAss1DProblem
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.helpers import HyperVolumeCalculator, compute_hull
from morlbench.plotting_stuff import plot_hypervolume



class TestAgents(unittest2.TestCase):

    def setUp(self):
        # create Problem
        self.gridworldproblem = MORLBuridansAssProblem()
        self.problem = Deepsea()
        # create an initialize randomly a weight vector
        self.scalarization_weights = np.zeros(self.problem.reward_dimension)
        self.scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)],
                                                   len(self.scalarization_weights))
        # tau is for chebyshev agent
        self.tau = 4.0
        # ref point is used for Hypervolume calculation
        self.ref = [-1.0, -1.0, -1.0]
        # learning rate
        self.alf = 0.1
        self.alfacheb = 0.1
        self.alfahvb = 0.1
        # Propability of epsilon greedy selection
        self.eps = 0.1
        # create one agent using chebyshev scalarization method
        self.chebyagent = MORLScalarizingAgent(self.gridworldproblem, [1.0, 0.0, 0.0], alpha=self.alfacheb, epsilon=self.eps,
                                               tau=self.tau, ref_point=self.ref)
        # create one agent using Hypervolume based Algorithm
        self.hvbagent = MORLHVBAgent(self.gridworldproblem, alpha=self.alfahvb, epsilon=self.eps, ref=self.ref,
                                     scal_weights=[1.0, 10.0])
        self.hagent = MORLHLearningAgent(self.problem, self.eps, self.alf, self.scalarization_weights)
        # both agents interact (times):
        self.interactions = 200
        self.convHullAgent = MORLConvexHullValueIteration(self.problem)
        self.data = [[4, 2, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1]]


class TestLearning(TestAgents):

    def runTest(self):
        self.runInteractions()
        self.runSelection()
        self.testWeightVariation()

    def runSelection(self):
        # make just one decision for each learned agent, to see if they act the same way
        new_state = self.chebyagent.decide(0, 3)
        print 'TEST(cheby): decision-from state 3-action:'+str(new_state)

        new_state2 = self.hvbagent.decide(0, 3)
        print 'TEST(hvb): decision-from state 3-action:'+str(new_state2)

    def runInteractions(self):
        # make the interactions
        p,m,s = morl_interact_multiple_episodic(self.hagent, self.problem, self.interactions)
        payouts, moves, states = morl_interact_multiple_episodic(self.chebyagent, self.gridworldproblem, self.interactions,
                                                                 max_episode_length=150)
        print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
              str(states[:]) + '\n')

        payouts2, moves2, states2 = morl_interact_multiple_episodic(self.hvbagent, self.gridworldproblem, self.interactions,
                                                                    max_episode_length=150)
        print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
              str(states2[:]) + '\n')

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
        self.agents.append(MORLScalarizingAgent(self.gridworldproblem, [1.0, 0.0, 0.0], alpha=self.alf, epsilon=self.eps,
                                                tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLScalarizingAgent(self.gridworldproblem, [0.0, 1.0, 0.0], alpha=self.alf, epsilon=self.eps,
                                                tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLScalarizingAgent(self.gridworldproblem, [0.5, 0.5, 0.0], alpha=self.alf, epsilon=self.eps,
                                                tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLScalarizingAgent(self.gridworldproblem, [0.0, 0.5, 0.5], alpha=self.alf, epsilon=self.eps,
                                                tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLScalarizingAgent(self.gridworldproblem, [0.5, 0.0, 0.5], alpha=self.alf, epsilon=self.eps,
                                                tau=self.tau, ref_point=self.ref))
        self.agents.append(MORLScalarizingAgent(self.gridworldproblem, [0.33, 0.33, 0.33], alpha=self.alf,
                                                epsilon=self.eps, tau=self.tau, ref_point=self.ref))

        # interact with each
        # for agent in self.agents:
        #   p, a, s = morl_interact_multiple_episodic(agent, self.gridworldproblem, self.interactions,
        #                                          max_episode_length=150)

        # plot_hypervolume(self.agents, self.agents[0]._morl_problem, name='weights')


class TestConvexHullValueIt(TestAgents):
    def runTest(self):
        self.runConvexHullIndices()


    def runConvexHullIndices(self):
        data = np.zeros((70, 2))
        for i in range(70):
            for u in range(2):
                data[i, u] = random.random()
        data = [[0.9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.0, 0.0, 0.0]]
        hull = compute_hull(data)
        plt.figure()
        for i in xrange(len(data)):
            plt.plot(data[i][0], data[i][1], 'ro')
        for q in xrange(len(hull)):
            plt.plot(hull[q][0], hull[q][1], 'bo')
        plt.axis([0, 2, 0, 2])
        plt.show()

    def testGetConvHull(self):
        data = self.data
        self.convHullAgent.get_hull(data)


class TestHyperVolumeCalculator(unittest2.TestCase):
    def setUp(self):
        # create refpoints
        self.ref_point2d = [0.1, 0.1]
        self.ref_point3d = [0.1, 0.1, 0.1]
        # data set / random points between 0/0 - 1/1
        self.set2d = np.zeros((70, 2))
        self.set3d = np.zeros((100, 3))
        for i in range(70):
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
        plt.figure()
        plt.axis([0-0.1, max(self.set2d[:, 0]*1.21), 0-0.1, max(self.set2d[:, 1]*1.1)])
        pfx = [pf[i][0] for i in range(len(pf))]
        pfy = [pf[u][1] for u in range(len(pf))]
        maxx = [max(pfx)]
        maxx.extend(pfx)
        pfx = maxx
        miny = [0]
        miny.extend(pfy)
        pfy = miny
        minx = 0
        pfx.append(minx)
        pfy.append(max(pfy))
        plt.plot(self.set2d[:, 0], self.set2d[:, 1], 'ro')
        plt.plot(pfx, pfy, 'bo', linestyle='--', drawstyle='steps-post')

        plt.fill_betweenx(pfx, 0, pfy, facecolor='blue', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('')
        plt.grid(False)
        filename = "pareto_front_2d.pdf"
        #plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()
        plt.subplot(131)
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

    def runCompute(self):
        hv = self.hv_2d_calc.compute_hv(self.set2d)
        hv3d = self.hv_3d_calc.compute_hv(self.set3d)
        print hv
        print hv3d


class TestProblems(unittest2.TestCase):
    def setUp(self):
        self.buridansassproblem = MORLBuridansAssProblem()
        self.buridansass1Dproblem = MORLBuridansAss1DProblem()
        self.puddleworldproblem = MOPuddleworldProblem()
        self.resourcegatheringproblem = MORLResourceGatheringProblem()
        self.mountaincarproblem = MountainCar()


class TestBuridan(TestProblems):
    def runTest(self):
        self.testDistance()
        self.testReward()
        self.testPlay()
        self.testFoodstolen()

    def testBuridanScene(self):
        plt.subplot(132)
        fig, ax = plt.subplots()
        scene = self.buridansassproblem._scene

        ax.imshow(scene, interpolation='nearest')
        step = 1.
        min = 0.
        rows = scene.shape[0]
        columns = scene.shape[1]
        row_arr = np.arange(min, rows)
        col_arr = np.arange(min, columns)
        x, y = np.meshgrid(row_arr, col_arr)
        for col_val, row_val in zip(x.flatten(), y.flatten()):
            c = int(scene[row_val, col_val])
            ax.text(col_val, row_val, c, va='center', ha='center')

    def testDistance(self):
        # tests if the distance is calculated correctly
        i = 1, 1
        u = 2, 1
        dist = self.buridansassproblem._get_distance(i, u)
        self.assertEqual(dist, 1.0, 'wrong distance')

    def testReward(self):
        # tests if the reward of staying in the same position gives zero reward
        self.buridansassproblem.reset()
        reward = self.buridansassproblem.play(0)
        reward = tuple([e for e in reward])
        self.assertTupleEqual(reward, (0, 0, 0), 'wrong reward')

    def testPlay(self):
        # tests if we get negative hunger reward if we go on 10 steps without eating
        self.buridansassproblem.reset()
        order = [1, 2, 3, 4, 4, 3, 2, 2, 1, 2]
        for i in order:
            r = self.buridansassproblem.play(i)
        reward = r[2]
        self.assertEqual(reward, -1.0, 'reward of hunger doesn\'t fit')

    def testFoodstolen(self):
        # sets the stealing probability to 100% and tests if the food is stolen
        # before set the problem to init state (middle)
        self.buridansassproblem.reset()
        temp = self.buridansassproblem.steal_probability
        self.buridansassproblem.steal_probability = 1
        # go upwards, beacause we get away from bottom right food visible neighborhood
        reward = self.buridansassproblem.play(2)
        # reward for stealing is second dimension
        steal_rew = reward[1]
        self.assertEqual(steal_rew, -0.5, 'no stealing reward')

    def testPrint(self):
        self.buridansassproblem.reset()
        # go down
        self.buridansassproblem.play(4)
        # check on map if he's gone down
        self.buridansassproblem.print_map(self.buridansassproblem._get_position(self.buridansassproblem.state))

    def testFoodEaten(self):
        self.buridansassproblem.reset()
        # go upwards
        self.buridansassproblem.play(2)
        # and then left
        self.buridansassproblem.play(3)
        # and stay there, to get food reward
        reward = self.buridansassproblem.play(0)[0]
        self.assertEqual(reward, 1.0, 'got wrong food reward')


class TestBuridan1D(TestProblems):
    def runTest(self):
        self.testDistance()
        self.testReward()
        self.testPlay()
        self.testFoodstolen()

    def testBuridanScene(self):
        plt.subplot(132)
        fig, ax = plt.subplots()
        scene = self.buridansass1Dproblem._scene

        ax.imshow(scene, interpolation='nearest')
        step = 1.
        min = 0.
        rows = scene.shape[0]
        columns = scene.shape[1]
        row_arr = np.arange(min, rows)
        col_arr = np.arange(min, columns)
        x, y = np.meshgrid(row_arr, col_arr)
        for col_val, row_val in zip(x.flatten(), y.flatten()):
            c = int(scene[row_val, col_val])
            ax.text(col_val, row_val, c, va='center', ha='center')

    def testDistance(self):
        # tests if the distance is calculated correctly
        i = 1, 1
        u = 2, 1
        dist = self.buridansass1Dproblem._get_distance(i, u)
        self.assertEqual(dist, 1.0, 'wrong distance')

    def testReward(self):
        # tests if the reward of staying in the same position gives zero reward
        self.buridansass1Dproblem.reset()
        reward = self.buridansass1Dproblem.play(0)
        reward = tuple([e for e in reward])
        self.assertTupleEqual(reward, (0, 0, 0), 'wrong reward')

    def testPlay(self):
        # tests if we get negative hunger reward if we go on 10 steps without eating
        self.buridansass1Dproblem.reset()
        order = [1, 2, 3, 4, 4, 3, 2, 2, 1, 2]
        for i in order:
            r = self.buridansass1Dproblem.play(i)
        reward = r[0]
        self.assertEqual(reward, -1.0, 'reward of hunger doesn\'t fit')

    def testFoodstolen(self):
        # sets the stealing probability to 100% and tests if the food is stolen
        # before set the problem to init state (middle)
        self.buridansass1Dproblem.reset()
        temp = self.buridansass1Dproblem.steal_probability
        self.buridansass1Dproblem.steal_probability = 1
        # go upwards, beacause we get away from bottom right food visible neighborhood
        reward = self.buridansass1Dproblem.play(2)
        # reward for stealing is second dimension
        steal_rew = reward[1]
        self.assertEqual(steal_rew, -0.5, 'no stealing reward')

    def testPrint(self):
        self.buridansass1Dproblem.reset()
        # go down
        self.buridansass1Dproblem.play(4)
        # check on map if he's gone down
        self.buridansass1Dproblem.print_map(self.buridansass1Dproblem._get_position(self.buridansass1Dproblem.state))

    def testFoodEaten(self):
        self.buridansass1Dproblem.reset()
        # go upwards
        self.buridansass1Dproblem.play(2)
        # and then left
        self.buridansass1Dproblem.play(3)
        # and stay there, to get food reward
        reward = self.buridansass1Dproblem.play(0)[0]
        self.assertEqual(reward, 1.0, 'got wrong food reward')


class TestPuddleworld(TestProblems):
    def runTest(self):
        self.testPuddleScene()
        self.testPlay()
        self.testPuddleReward()

    def testPuddleScene(self):
        plt.subplot(141)
        fig, ax = plt.subplots()
        scene = self.puddleworldproblem._scene

        ax.imshow(scene, interpolation='nearest')
        step = 1.
        min = 0.
        rows = scene.shape[0]
        columns = scene.shape[1]
        row_arr = np.arange(min, rows)
        col_arr = np.arange(min, columns)
        x, y = np.meshgrid(row_arr, col_arr)
        for col_val, row_val in zip(x.flatten(), y.flatten()):
            c = int(scene[row_val, col_val])
            ax.text(col_val, row_val, c, va='center', ha='center')
        plt.show()

    def testPlay(self):
        self.puddleworldproblem.reset()
        order = [1]
        for i in order:
            r = self.puddleworldproblem.play(i)
        reward = r[0]
        self.assertEqual(reward, -1.0, 'reward of going doesn\'t fit')

    def testPuddleReward(self):
        self.puddleworldproblem.reset()
        self.puddleworldproblem.state = 28

        r = self.puddleworldproblem.play(1)
        touched = r[1]
        r = self.puddleworldproblem.play(1)
        touched2 = r[1]
        t = touched, touched2
        self.assertTupleEqual(t, (-20.0, -40.0), 'puddlereward wrong')

    def testFinalReward(self):
        self.puddleworldproblem.reset()
        self.puddleworldproblem.state = 17

        self.puddleworldproblem.play(0)
        reward = self.puddleworldproblem.play(0)
        self.assertEqual(reward[0], 10.0, 'final reward wrong')


class TestResourceGathering(TestProblems):
    def runTest(self):
        self.testResScene()
        self.testResources()
        self.testResourceReward()

    def testResScene(self):
        self.resourcegatheringproblem.__init__()

        fig, ax = plt.subplots()
        scene = self.resourcegatheringproblem._scene
        scene[4, 2] = 9
        ax.imshow(scene, interpolation='None')
        step = 1.
        min = 0.
        rows = scene.shape[0]
        columns = scene.shape[1]
        row_arr = np.arange(min, rows)
        col_arr = np.arange(min, columns)
        x, y = np.meshgrid(row_arr, col_arr)
        for col_val, row_val in zip(x.flatten(), y.flatten()):
            c = int(scene[row_val, col_val])
            if col_val == 2 and row_val == 4:
                c = "H"
            ax.text(col_val, row_val, c, va='center', ha='center')
        plt.show()

    def testPosition(self):
        for i in xrange(10):
            ind = random.choice(np.arange(self.resourcegatheringproblem.n_states))
            print str(ind) + ': ' + str(self.resourcegatheringproblem._get_position(ind))
        self.resourcegatheringproblem.state = 99
        self.resourcegatheringproblem.play(3)

    def testResources(self):
        self.resourcegatheringproblem.reset()
        # run to first resource
        order = [2, 1, 1, 1, 1, 0]
        for i in order:
            r = self.resourcegatheringproblem.play(i)
        state = self.resourcegatheringproblem.state
        r1 = self.resourcegatheringproblem._get_position(state)[2]
        self.resourcegatheringproblem.reset()
        # run to second resource:
        order = [1, 0, 1, 1, 0]
        for i in order:
            self.resourcegatheringproblem.play(i)
        state = self.resourcegatheringproblem.state
        r2 = self.resourcegatheringproblem._get_position(state)[3]

        self.assertEqual(r1, 1, 'first resource not found')
        self.assertEqual(r2, 1, 'second resource not found')

    def testWorld3d(self):
        state = self.resourcegatheringproblem.state
        for i in xrange(self.resourcegatheringproblem.n_states):
            position = self.resourcegatheringproblem._get_position(i)
            state = self.resourcegatheringproblem._get_index(position)
            self.assertEqual(state, i, 'index %d is wrong' % i)
        print self.resourcegatheringproblem._get_index((1, 4, 0, 0))

    def testResourceReward(self):
        self.resourcegatheringproblem.reset()
        # run to first resource
        order = [2, 1, 1, 1, 1, 0]
        for i in order:
            r = self.resourcegatheringproblem.play(i)
        order = [2, 3, 3, 3, 3, 0]
        for i in order:
            r = self.resourcegatheringproblem.play(i)
        # look wether we got a reward
        rew1 = r[1]
        self.resourcegatheringproblem.reset()
        # run to second resource:
        order = [1, 0, 1, 1, 0]
        for i in order:
            self.resourcegatheringproblem.play(i)
        order = [3, 3, 3, 2, 2]
        for i in order:
            r =self.resourcegatheringproblem.play(i)
        rew2 = r[2]

        self.assertEqual(rew1, 1, 'no reward for first resource')
        self.assertEqual(rew2, 1, 'no reward for second resource')

    def testEnemyReward(self):
        self.resourcegatheringproblem.reset()
        self.resourcegatheringproblem.losing_probability=1
        order = [1, 1, 1]
        for i in order:
            r = self.resourcegatheringproblem.play(i)

        s = self.resourcegatheringproblem.state
        rew = r[0]
        bag_size = self.resourcegatheringproblem._bag.count(1)

        self.assertEqual(s, self.resourcegatheringproblem.init, 'did not set the player on init position after losing')
        self.assertEqual(rew, -10.0, 'did not get negative reward after losing')
        self.assertEqual(bag_size, 0, 'bag wasn\'t emptied after losing against enemy')

