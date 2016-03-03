"""
Created on Feb 16, 2016

@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
"""

import cPickle as pickle
from morl_problems import Deepsea
import time

# import plotly.plotly as py
# import plotly.graph_objs as go

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import numpy as np

from morl_policies import *

def heatmap_matplot(problem, states):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig.suptitle('Heatmap Plot of visited States', fontsize=14, fontweight='bold')

    _heatmap_matplot(problem, states, ax)

    plt.show()


def _heatmap_matplot(problem, states, ax):
    """
    Plots a simple heatmap of visited states for a given problem (e.g. Deepsea)
     and an array of recorded states.
    :param problem: problem object (e.g. Deepsea)
    :param states: array of states per episode
    :return: shows a plot
    """

    x_dim, y_dim = problem.scene_x_dim, problem.scene_y_dim

    # Initialization of empty heatmap (-1 -> terminal state)
    heatmap = np.zeros(problem.n_states - 1)

    # Count states per episode and sum them up
    for i in xrange(states.size):
        z = np.bincount(states[i])
        heatmap[:len(z)] += z

    # Shape the linearized heatmap according to the problem geometry
    heatmap_shaped = heatmap.reshape(problem._scene.shape)

    mycmap = plt.get_cmap("YlOrRd")
    norm = colors.Normalize(vmin=min(heatmap), vmax=max(heatmap))

    for y in xrange(y_dim):
        for x in xrange(x_dim):
            ax.scatter(x, -y, s=100, color=mycmap(norm(heatmap_shaped[y, x])), cmap=mycmap)

            if problem._scene[y, x] < 0:
                ax.add_patch(patches.Rectangle((x-0.5, -y-0.5), 1, 1, facecolor='black'))

            if problem._scene[y, x] > 0:
                ax.add_patch(patches.Rectangle((x-0.5, -y-0.5), 1, 1,
                                               facecolor='none', edgecolor='red', linestyle='dotted'))

    ax.add_patch(patches.Rectangle((-0.5, -y_dim+0.5), x_dim, y_dim, facecolor='none', lw=2))

    def format_coord(x, y):
        x = int(x)
        y = int(abs(-y))
        y = int(abs(-y))
        if x >= 0 and x <= x_dim-1 and x >= 0 and y <= y_dim-1:
            z = heatmap_shaped[y, x]
            return 'x=%1.0f, y=%1.0f, z=%1.2f' % (x, y, z)
        else:
            return 'x=%1.0f, y=%1.0f' % (x, y)

    ax.format_coord = format_coord

    ticks_offset = 1
    plt.yticks(range(-y_dim+ticks_offset,0+ticks_offset),
               tuple(map(str, range(y_dim-ticks_offset, 0-ticks_offset, -1))))

    plt.xticks(range(0, x_dim),
               tuple(map(str, range(0, x_dim, 1))))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top') # the rest is the same

    ax.margins(0.0)

    return ax




def _policy_plot2(problem, policy, ax):

    x_dim, y_dim = problem.scene_x_dim, problem.scene_y_dim

    _pi = policy.get_pi()

    mycmap = plt.get_cmap("YlOrRd")
    #norm = colors.Normalize(vmin=min(heatmap), vmax=max(heatmap))

    for y in xrange(y_dim):
        for x in xrange(x_dim):
            if problem._scene[y,x] < 0:
                ax.add_patch(patches.Rectangle((x-0.5, -y-0.5), 1, 1, facecolor='black'))
            elif problem._scene[y,x] > 0:
                ax.add_patch(patches.Rectangle((x-0.5, -y-0.5), 1, 1,
                                               facecolor='none', edgecolor='red', linestyle='dotted'))
                ax.annotate(problem._scene[y,x], (x, -y), color='black', weight='bold',
                fontsize=12, ha='center', va='center')
            else:
                for a in xrange(problem.n_actions-1):
                    off1 = problem.actions[a] * 0.15
                    off2 = problem.actions[a] * 0.23
                    ax.add_patch(patches.FancyArrow(x+off1[1], -y-off1[0], off2[1], -off2[0],
                                                    width=0.3, head_width=0.3, head_length=0.1, lw=0,
                                                    fc=mycmap(_pi[problem._get_index((y, x)), a])))

    ax.add_patch(patches.Rectangle((-0.5, -y_dim+0.5), x_dim, y_dim, facecolor='none', lw=2))

    def format_coord(x, y):
        x = int(x)
        y = int(abs(-y))
        y = int(abs(-y))
        if x >= 0 and x <= x_dim-1 and x >= 0 and y <= y_dim-1:
            for a in xrange(problem.n_actions-1):
                vals = _pi[problem._get_index((y, x)), :]
                vals = ' '.join('%1.2f' % v for v in vals )
            return 'x=%1.0f, y=%1.0f, values=%s' % (x, y, vals)
        else:
            return 'x=%1.0f, y=%1.0f' % (x, y)

    ax.format_coord = format_coord

    ticks_offset = 1
    plt.yticks(range(-y_dim+ticks_offset,0+ticks_offset),
               tuple(map(str, range(y_dim-ticks_offset, 0-ticks_offset, -1))))

    plt.xticks(range(0, x_dim),
               tuple(map(str, range(0, x_dim, 1))))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top') # the rest is the same
    ax.margins(0.0)

    return ax


def policy_plot2(problem, policy, title=None, filename=None):


    fig = plt.figure()

    ax = fig.add_subplot(111)

    _policy_plot2(problem, policy, ax)

    if not filename:
        filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S" + '.pdf')
    else:
        filename += '_' + time.strftime("%H%M%S") + '.pdf'

    if not title:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        title = 'Policy Plot'
    else:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()


def policy_heat_plot(problem, policy, states, title=None, filename=None):


    fig = plt.figure()

    ax = fig.add_subplot(111)

    _heatmap_matplot(problem, states, ax)

    _policy_plot2(problem, policy, ax)

    # plt.figure(fig1.number)
    # fig1.sca(plt.gca())
    # plt.axes()
    # fig2.show()
    plt.show()


def policy_plot(problem, policy, filename=None):
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

    if not(filename):
        filename = 'figure_' + time.strftime("%Y%m%d-%H%M%S")
    plt.title('Policy Plot')
    plt.savefig(filename, bbox_inches='tight')

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
    plt.title('Transition Plot')
    plt.show()

def heatmap_plotly():
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
    pass

if __name__ == '__main__':
    problem = Deepsea()

    payouts, moves, states, problem, agent = pickle.load(open("scalQ_e0.7a0.3W=[1.0, 0.0]_114301.p"))

    #policy = PolicyDeepseaExpert(problem, task='T2')

    policy = PolicyDeepseaFromAgent(problem, agent)

    # policy_plot2(problem, policy)
    # heatmap_matplot(problem, states)
    policy_heat_plot(problem, policy, states)

