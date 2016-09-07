from morlbench.morl_problems import MORLResourceGatheringProblem, MountainCar, MORLGridworld, MORLBuridansAss1DProblem, \
        Deepsea, MOPuddleworldProblem
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot, policy_plot2
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt
import logging as log

if __name__ == '__main__':
    # create Problem
    problem = MORLGridworld()
    random.seed(2)
    np.random.seed(2)
    # learning rate
    alfacheb = 0.11
    eps = 0.9
    ref_points = [[10.0, -1000.0, 10.0], [-1000.0, 10.0, 10.0], [10.0, 10.0,  -1000.0]]
    agents = []
    scalarization_weights = [0.0, 0.0]
    interactions = 1000
    log.info('Started reference point experiment')
    payoutslist = []
    for ref_p in xrange(len(ref_points)):
        agents.append(MORLHVBAgent(problem, alfacheb, eps, ref_points[ref_p], scalarization_weights))

        payouts, moves, states = morl_interact_multiple_episodic(agents[ref_p], problem, interactions, max_episode_length=300)
        payoutslist.append(payouts)
        policy = PolicyFromAgent(problem, agents[ref_p], mode='greedy')
        # policy_heat_plot(problem, policy, states)

    plot_hypervolume(agents, problem, name='reference point')
    print 'final average reward' + str(np.mean(payoutslist[0], axis=0))
    print 'final average reward' + str(np.mean(payoutslist[1], axis=0))
    print 'final average reward' + str(np.mean(payoutslist[2], axis=0))


