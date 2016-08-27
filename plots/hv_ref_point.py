from morlbench.morl_problems import MORLResourceGatheringProblem, MountainCarTime, MORLGridworld, MORLBuridansAss1DProblem, \
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
    # learning rate
    alfacheb = 0.11
    eps = 0.9
    ref_points = [[-0.1, -100.0, -0.1], [-100.0, -0.1, -0.1], [-0.1, -0.1, -100.0]]
    agents = []
    scalarization_weights = [0.0, 0.0]
    interactions = 8000
    log.info('Started reference point experiment')
    payoutslist = []
    for ref_p in xrange(len(ref_points)):
        agents.append(MORLHVBAgent(problem, alfacheb, eps, ref_points[ref_p], scalarization_weights))

        payouts, moves, states = morl_interact_multiple_episodic(agents[ref_p], problem, interactions)
        payoutslist.append(payouts)
        policy = PolicyFromAgent(problem, agents[ref_p], mode='greedy')
        # policy_heat_plot(problem, policy, states)

    plot_hypervolume(agents, problem, name='reference point')
    print 'final average reward' + str(sum(payoutslist[0]))
    print 'final average reward' + str(sum(payoutslist[1]))
    print 'final average reward' + str(sum(payoutslist[2]))


