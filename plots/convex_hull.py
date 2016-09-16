import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from scipy.spatial import ConvexHull
if __name__ == '__main__':
    mpl.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    count = 20
    random.seed(18.9654)
    ref_point2d = [0.001, 0.001]
    set2d = np.zeros((count, 2))
    for i in range(count):
        for u in range(2):
            rand = random.random()
            set2d[i, u] = rand if (rand > ref_point2d)or(rand > 0.3) else random.random()
    #hv_2d_calc = HyperVolumeCalculator(ref_point2d)
    #pf = hv_2d_calc.extract_front(set2d)
    hullraw = ConvexHull(set2d)
    hull = [set2d[x] for x in hullraw.vertices]
    hull.append(hull[0])
    size = 0.48 * 5.8091048611149611602
    fig = plt.figure(figsize=[size, 0.75 * size])

    fig.set_size_inches(size, 0.7 * size)
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    plt.axis([0, max(set2d[:, 0]+0.07), 0.05, max(set2d[:, 1]*1.1)])
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    pfx = [hull[i][0] for i in range(len(hull))]
    pfy = [hull[u][1] for u in range(len(hull))]
    plt.plot(set2d[:, 0], set2d[:, 1], 'bo', markersize=4)
    plt.plot(pfx, pfy, 'ro', markersize=4)
    plt.fill(pfx, pfy, facecolor='red', alpha=0.3)
    plt.xlabel('Ziel 1', size=9)
    plt.ylabel('Ziel 2', size=9)
    plt.title('')
    plt.grid(False)
    plt.subplots_adjust(bottom=0.18, left=0.17)

    plt.show()