from morlbench.morl_problems import  Deepsea, MOPuddleworldProblem, MORLBuridansAssProblem, MORLGridworld, MORLResourceGatheringProblem
from morlbench.morl_agents import MORLChebyshevAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt
import logging as log
"""
This experiment shows the use of the hypervolume metric as quality indicator
There are two parts of the experiment:
    - It takes one Problem and lets train two different agents (1) upon it.
    - It takes one Problem and lets train six equal agents with different weights (2) on it
At the end it shows the evolution in two seperate plots
You can:
    - change reference point
    - play with the chebishev learning parameter tau
    - change weights of chebishev agent
    - play with learning rates
    - alternate epsilon
    - train more or less often by adjusting interactions
    and see what happens to the learning process
Attention:
    - High epsilons may slow down learning process
    - Too small learning rates cause little impact and small learning effect
    - Too big learning rates cause too big impact on learning process
    - Sum of weight vector elements should equal 1
    - learning rate alfa, epsilon and lambda are parameters out of [0, 1]
"""
if __name__ == '__main__':
    # create Problem
    problem = MORLBuridansAssProblem()
    # create an initialize randomly a weight vector
    scalarization_weights = np.zeros(problem.reward_dimension)
    scalarization_weights = random.sample([i for i in np.linspace(0, 5, 5000)], len(scalarization_weights))
    # tau is for chebyshev agent
    tau = 50.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, -1.0, -1.0]
    # learning rate
    alf = 0.2
    alfacheb = 0.05
    alfahvb = 0.01

    # Propability of epsilon greedy selection
    eps = 0.1
    # create one agent using chebyshev scalarization method
    chebyagent = MORLChebyshevAgent(problem, [1.0, 0.0, 0.0], alpha=alfacheb, epsilon=eps,
                                    tau=tau, ref_point=ref)
    # create one agent using Hypervolume based Algorithm
    hvbagent = MORLHVBAgent(problem, alpha=alfahvb, epsilon=0.6, ref=ref, scal_weights=[1.0, 10.0])
    # both agents interact (times):
    interactions = 1000
    # make the interactions
    log.info('Playing %i interactions on chebyagent' % interactions)
    payouts, moves, states = morl_interact_multiple(chebyagent, problem, interactions,
                                                    max_episode_length=150)
    # print("TEST(cheby): interactions made: \nP: "+str(payouts[:])+",\n M: " + str(moves[:]) + ",\n S: " +
    #     str(states[:]) + '\n')
    log.info('Playing %i interactions on hvb agent' % interactions)
    payouts2, moves2, states2 = morl_interact_multiple(hvbagent, problem, interactions,
                                                       max_episode_length=150)
    # print("TEST(HVB): interactions made: \nP: "+str(payouts2[:])+",\n M: " + str(moves2[:]) + ",\n S: " +
    #      str(states2[:]) + '\n')

    # extract all volumes of each agent
    agents = [hvbagent, chebyagent]

    # list of agents with different weights
    agent_group = []
    # list of volumes
    vollist = []
    # 6 agents with each different weights
    weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5],
               [0.5, 0.0, 0.5], [0.33, 0.33, 0.33]]
    for weight in weights:
        agent_group.append(MORLChebyshevAgent(problem, weight, alpha=alfacheb, epsilon=eps,
                                               tau=tau, ref_point=ref))

    # interact with each
    log.info('Playing %i interactions on %i chebyagents' % interactions, len(agents))
    for agent in agent_group:
        p, a, s = morl_interact_multiple(agent, problem, interactions,
                                         max_episode_length=150)
    # plot the evolution of hv of every weights agent
    plot_hypervolume(agent_group, agent_group[0]._morl_problem, name='weights')

    # plot the evolution of both agents hypervolume metrics
    plot_hypervolume(agents, problem)

