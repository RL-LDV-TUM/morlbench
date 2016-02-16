"""
Created on Feb 16, 2016

@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
"""

import cPickle as pickle
from morl_problems import Deepsea

import plotly.plotly as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

def heatmap_matplot():
    fig, ax = plt.subplots()

    colormap = cm.jet # color map

    colormap.set_bad(color='grey') # set color for mask (ground)

    ax.imshow(heatmap_mask, colormap, interpolation='nearest')

    numrows, numcols = heatmap.shape

    # Move ticks to the middle of each field
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = heatmap[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord


    # X, Y = np.meshgrid(np.arange(0, -10, -1), np.arange(0, 9, 1))
    # U = np.cos(X)
    # V = np.sin(Y)
    # Q = plt.quiver(U, V)

    plt.show()

def transition_map():
    plot_map = np.zeros((3*problem.scene_y_dim, 3*problem.scene_x_dim))

    for i in xrange(non_zero_states.size):
        coords = problem._get_position(non_zero_states[i])

        plot_map[coords[0]*3][(coords[1]*3)+1] = transition_probabilities[0][i][0] # first action (up)
        plot_map[(coords[0]*3)+2][(coords[1]*3)+1] = transition_probabilities[0][i][1] # second action (down)
        plot_map[(coords[0]*3)+1][(coords[1]*3)] = transition_probabilities[0][i][2] # first action (right)
        plot_map[(coords[0]*3)+1][(coords[1]*3)+2] = transition_probabilities[0][i][3] # first action (left)

    trans_map_masked = np.repeat(problem._scene, 3, axis=1)
    trans_map_masked = np.repeat(trans_map_masked, 3, axis=0)
    trans_map_masked = np.ma.masked_where(trans_map_masked == -100, trans_map_masked)

    trans_map_masked[trans_map_masked >= 0] = plot_map[trans_map_masked >= 0]

    fig, ax = plt.subplots()
    colormap = cm.jet # color map
    colormap.set_bad(color='grey') # set color for mask (ground)
    ax.imshow(trans_map_masked, colormap, interpolation='nearest')
    numrows, numcols = trans_map_masked.shape


    plt.xticks( np.arange(1,30,3), ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9') )
    plt.yticks( np.arange(1,33,3), ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10') )

    for i in xrange(len(y)):
        ax.axhline(3*i, linestyle='--', color='k')

    for i in xrange(len(x)):
        ax.axvline(3*i, linestyle='--', color='k')

    # Move ticks to the middle of each field
    # def format_coord(x, y):
    #     col = int(x + 1.5)
    #     row = int(y + 1.5)
    #     if col >= 0 and col < numcols and row >= 0 and row < numrows:
    #         z = trans_map_masked[row, col]
    #         return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    #     else:
    #         return 'x=%1.4f, y=%1.4f' % (x, y)

    # ax.format_coord = format_coord
    plt.show()

def heatmap_plotly():
    py.sign_in('xtra', 'jut0nmg713')
    annotations = []
    for n, row in enumerate(heatmap):
        for m, val in enumerate(row):
            var = heatmap[n][m]
            annotations.append(
                dict(
                    text=str(val),
                    x=x[m], y=y[n],
                    xref='x1', yref='y1',
                    font=dict(color='white' if val > 0.5 else 'black'),
                    showarrow=False)
                )

    colorscale = [[0, '#3D9970'], [1000, '#001f3f']]  # custom colorscale
    trace = go.Heatmap(x=x, y=y, z=heatmap, colorscale=colorscale, showscale=False)

    fig = go.Figure(data=[trace])
    fig['layout'].update(
        title="Policy Heatmap",
        annotations=annotations,
        xaxis=dict(ticks='', side='top'),
        # ticksuffix is a workaround to add a bit of padding
        yaxis=dict(ticks='', ticksuffix='  '),
        width=700,
        height=700,
        autosize=False
    )
    url = py.plot(fig, filename='Annotated Heatmap', height=750)

if __name__ == '__main__':
    problem = Deepsea()
    payouts, moves, states = pickle.load(open("results_10000.p"))

    heatmap = np.zeros(problem.n_states)

    policy = np.zeros((problem.n_states,problem.n_actions))

    for i in xrange(states.size):
        z = np.bincount(states[i])
        heatmap[:len(z)] += z
        for j in xrange(len(states[i])):
            policy[states[i][j]][moves[i][j]] += 1

    non_zero_states = np.where(heatmap > 0)[0]
    transition_probabilities = policy[np.nonzero(heatmap)] / heatmap[np.nonzero(heatmap), None]

    heatmap = heatmap.reshape(problem._scene.shape)

    # Generate masked heatmap (ground is masked for plotting)
    heatmap_mask = np.ma.masked_where(problem._scene == -100, heatmap)

    x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    y = ['-0', '-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9', '-10']

    transition_map()
    #heatmap_matplot()
    #heatmap_plotly()




