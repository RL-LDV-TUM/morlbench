"""
Created on Feb 16, 2016

@author: Johannes Feldmaier <johannes.feldmaier@tum.de>
"""

import cPickle as pickle
from morl_problems import Deepsea

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

py.sign_in('xtra', 'jut0nmg713')

problem = Deepsea()
payouts, moves, states = pickle.load(open("results.p"))

heatmap = np.zeros(problem.n_states)

for i in xrange(states.size):
    z = np.bincount(states[i])

    heatmap[:len(z)] += z

heatmap = heatmap.reshape(problem._scene.shape)

x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
y = ['-0', '-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9', '-10']


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

print payouts
print moves
print states
