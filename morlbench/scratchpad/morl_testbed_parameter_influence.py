from morlbench.morl_problems import MORLResourceGatheringProblem, MountainCar, MORLGridworld, MORLBuridansAssProblem, Deepsea
from morlbench.morl_agents import MORLScalarizingAgent, MORLHVBAgent
from morlbench.experiment_helpers import morl_interact_multiple_episodic
from morlbench.morl_policies import PolicyFromAgent
from morlbench.plot_heatmap import policy_heat_plot
from morlbench.plotting_stuff import plot_hypervolume

import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':

    epsilon_experiment = True
    gamma_experiment = False
    alpha_experiment = False
    # create Problem
    problem = Deepsea()
    # create an initialize randomly a weight vector
    scalarization_weights = [1.0, 0.0]
    # tau is for chebyshev agent
    tau = 4.0
    # ref point is used for Hypervolume calculation
    ref = [-1.0, ]*problem.reward_dimension
    # learning rate
    alfacheb = 0.11
    # Propability of epsilon greedy selection
    epsilons = np.arange(0, 1, 0.1)
    gammas = np.arange(0, 1, 0.1)
    alphas = np.arange(0, 1, 0.1)
    # agents:
    agents = []
    interactions = 600
    if epsilon_experiment:
        for eps in xrange(len(epsilons)):
            agents.append(MORLScalarizingAgent(problem, epsilon=epsilons[eps], alpha=alfacheb,
                                               scalarization_weights=scalarization_weights,
                                               ref_point=ref, tau=tau, function='chebishev'))
            morl_interact_multiple_episodic(agents[eps], problem, interactions)

        plot_hypervolume(agents, problem, name='epsilon')

    if gamma_experiment:
        for gam in xrange(len(gammas)):
            agents.append(MORLScalarizingAgent(problem, epsilon=0.1, alpha=alfacheb,
                                               scalarization_weights=scalarization_weights,
                                               ref_point=ref, tau=tau, function='chebishev', gamma=gammas[gam]))
            morl_interact_multiple_episodic(agents[gam], problem, interactions)

        plot_hypervolume(agents, problem, name='gamma')

    if alpha_experiment:
        for alf in xrange(len(alphas)):
            agents.append(MORLScalarizingAgent(problem, epsilon=0.1, alpha=alphas[alf],
                                               scalarization_weights=scalarization_weights,
                                               ref_point=ref, tau=tau, function='chebishev'))
            morl_interact_multiple_episodic(agents[alf], problem, interactions)

        plot_hypervolume(agents, problem, name='alpha')

