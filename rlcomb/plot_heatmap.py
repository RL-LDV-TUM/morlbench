"""
Created on Feb 16, 2016

@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
"""

import cPickle as pickle
from morl_problems import Deepsea

# import plotly.plotly as py
# import plotly.graph_objs as go

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np


def heatmap_matplot(problem, states):
    """
    Plots a simple heatmap of visited states for a given problem (e.g. Deepsea)
     and an array of recorded states.
    :param problem: problem object (e.g. Deepsea)
    :param states: array of states per episode
    :return: shows a plot
    """
    # Initialization of empty arrays
    heatmap = np.zeros(problem.n_states)

    # Count states per episode and sum them up
    for i in xrange(states.size):
        z = np.bincount(states[i])
        heatmap[:len(z)] += z

    # Shape the linearized heatmap according to the problem geometry
    heatmap = heatmap.reshape(problem._scene.shape)

    # Generate masked heatmap (ground is masked for plotting)
    heatmap_mask = np.ma.masked_where(problem._scene == -100, heatmap)

    fig, ax = plt.subplots()

    colormap = cm.jet  # color map

    colormap.set_bad(color='grey')  # set color for mask (ground)

    ax.imshow(heatmap_mask, colormap, interpolation='nearest')

    numrows, numcols = heatmap.shape

    # x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # y = ['-0', '-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9', '-10']

    # Set z-value to the heatmap value -> so it can be read in the plot
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = heatmap[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord

    plt.show()


def policy_plot(problem, policy):
    """
    Plot the transition probabilities for a specific policy.

    :param problem: problem object (e.g. Deepsea)
    :param policy: policy object.
    :return: noting
    """
    # Initialization of empty arrays
    x_dim, y_dim = problem.scene_x_dim, problem.scene_y_dim
    plot_map = np.zeros((3*y_dim, 3*x_dim))  # plot array with 3x3 pixel for each state
    transition_probabilities = policy.get_pi()[0:x_dim*y_dim]

    # Find non visited states
    non_zero_states = (problem._flat_map >= 0)

    # Place subpixel in each 3x3 state pixel
    for i in xrange(non_zero_states.size):
        if non_zero_states[i]:
            coords = problem._get_position(i)
            plot_map[coords[0]*3][(coords[1]*3)+1] = transition_probabilities[i, 0]  # first action (up)
            plot_map[(coords[0]*3)+2][(coords[1]*3)+1] = transition_probabilities[i, 1]  # second action (down)
            plot_map[(coords[0]*3)+1][(coords[1]*3)+2] = transition_probabilities[i, 2]  # first action (right)
            plot_map[(coords[0]*3)+1][(coords[1]*3)] = transition_probabilities[i, 3]  # first action (left)

    # repeat ground mask three times and map the values to the expanded mask
    trans_map_masked = np.repeat(problem._scene, 3, axis=1)
    trans_map_masked = np.repeat(trans_map_masked, 3, axis=0)
    trans_map_masked = np.ma.masked_where(trans_map_masked == -100, trans_map_masked)

    # Copy all values of plot_map to the masked map containing the ground profile
    trans_map_masked[trans_map_masked >= 0] = plot_map[trans_map_masked >= 0]

    # Generate the heatmap plot
    fig, ax = plt.subplots()
    colormap = cm.jet  # color map
    colormap.set_bad(color='grey')  # set color for mask (ground)
    ax.imshow(trans_map_masked, colormap, interpolation='nearest')
    numrows, numcols = trans_map_masked.shape

    xticks_labels = tuple(map(str, range(problem.scene_x_dim)))
    yticks_labels = tuple(map(str, range(problem.scene_y_dim)))

    plt.xticks(np.arange(1, 30, 3), xticks_labels)
    plt.yticks(np.arange(1, 33, 3), yticks_labels)

    for i in xrange(1, len(yticks_labels)):
        ax.axhline((3*i)-0.5, linestyle='--', color='k')

    for i in xrange(len(xticks_labels)):
        ax.axvline((3*i)-0.5, linestyle='--', color='k')

    # Set z-value to the heatmap value -> so it can be read in the plot
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = trans_map_masked[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord

    plt.show()


def transition_map(problem, states, moves):
    """
    Plots transition probabilities for a given problem and the
    recorded states and moves.
    The function is only tested with the Deepsea environment
    :param problem: problem object (e.g. Deepsea)
    :param states: array of states per episode
    :param moves: array of actions performed in each state for each episode
    :return: nothing - a plot is shown
    """
    # Initialization of empty arrays
    x_dim, y_dim = problem.scene_x_dim, problem.scene_y_dim
    plot_map = np.zeros((3*y_dim, 3*x_dim))  # plot array with 3x3 pixel for each state
    heatmap = np.zeros(problem.n_states - 1)
    policy = np.zeros((problem.n_states - 1, problem.n_actions))

    # Count states per episode and sum them up
    for i in xrange(states.shape[0]):
        z = np.bincount(states[i])
        heatmap[:len(z)] += z
        # Count actions per state for all episodes
        for j in xrange(len(states[i])):
            policy[states[i][j]][moves[i][j]] += 1

    # Find non visited states
    non_zero_states = np.where(heatmap > 0)[0]

    # Calculate probabilities for all non zero states
    transition_probabilities = policy[np.nonzero(heatmap)] / heatmap[np.nonzero(heatmap), None]

    # Place subpixel in each 3x3 state pixel
    for i in xrange(non_zero_states.size):
        coords = problem._get_position(non_zero_states[i])
        plot_map[coords[0]*3][(coords[1]*3)+1] = transition_probabilities[0][i][0]  # first action (up)
        plot_map[(coords[0]*3)+2][(coords[1]*3)+1] = transition_probabilities[0][i][1]  # second action (down)
        plot_map[(coords[0]*3)+1][(coords[1]*3)+2] = transition_probabilities[0][i][2]  # first action (right)
        plot_map[(coords[0]*3)+1][(coords[1]*3)] = transition_probabilities[0][i][3]  # first action (left)

    # repeat ground mask three times and map the values to the expanded mask
    trans_map_masked = np.repeat(problem._scene, 3, axis=1)
    trans_map_masked = np.repeat(trans_map_masked, 3, axis=0)
    trans_map_masked = np.ma.masked_where(trans_map_masked == -100, trans_map_masked)

    # Copy all values of plot_map to the masked map containing the ground profile
    trans_map_masked[trans_map_masked >= 0] = plot_map[trans_map_masked >= 0]

    # Generate the heatmap plot
    fig, ax = plt.subplots()
    colormap = cm.jet  # color map
    colormap.set_bad(color='grey')  # set color for mask (ground)
    ax.imshow(trans_map_masked, colormap, interpolation='nearest')
    numrows, numcols = trans_map_masked.shape

    xticks_labels = tuple(map(str, range(x_dim)))
    yticks_labels = tuple(map(str, range(y_dim)))

    plt.xticks(np.arange(1, 30, 3), xticks_labels)
    plt.yticks(np.arange(1, 33, 3), yticks_labels)

    for i in xrange(1, len(yticks_labels)):
        ax.axhline((3*i)-0.5, linestyle='--', color='k')

    for i in xrange(len(xticks_labels)):
        ax.axvline((3*i)-0.5, linestyle='--', color='k')

    # Set z-value to the heatmap value -> so it can be read in the plot
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = trans_map_masked[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord

    plt.show()

# def heatmap_plotly():
#     py.sign_in('xtra', 'jut0nmg713')
#     annotations = []
#     for n, row in enumerate(heatmap):
#         for m, val in enumerate(row):
#             var = heatmap[n][m]
#             annotations.append(
#                 dict(
#                     text=str(val),
#                     x=x[m], y=y[n],
#                     xref='x1', yref='y1',
#                     font=dict(color='white' if val > 0.5 else 'black'),
#                     showarrow=False)
#                 )
#
#     colorscale = [[0, '#3D9970'], [1000, '#001f3f']]  # custom colorscale
#     trace = go.Heatmap(x=x, y=y, z=heatmap, colorscale=colorscale, showscale=False)
#
#     fig = go.Figure(data=[trace])
#     fig['layout'].update(
#         title="Policy Heatmap",
#         annotations=annotations,
#         xaxis=dict(ticks='', side='top'),
#         # ticksuffix is a workaround to add a bit of padding
#         yaxis=dict(ticks='', ticksuffix='  '),
#         width=700,
#         height=700,
#         autosize=False
#     )
#     url = py.plot(fig, filename='Annotated Heatmap', height=750)

if __name__ == '__main__':
    prob = Deepsea()
    saved_payouts, saved_moves, saved_states = pickle.load(open("results_10000_eps0.8_0.5-0.5.p"))

    transition_map(problem=prob, states=saved_states, moves=saved_moves)
    # heatmap_matplot()
    # heatmap_plotly()




