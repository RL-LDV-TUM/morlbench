import numpy as np
import matplotlib.pyplot as plt
import random
from morlbench.helpers import HyperVolumeCalculator
from scipy.spatial import ConvexHull
if __name__ == '__main__':
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    plt.axis([0, max(set2d[:, 0]+0.07), 0.05, max(set2d[:, 1]*1.1)])
    pfx = [hull[i][0] for i in range(len(hull))]
    pfy = [hull[u][1] for u in range(len(hull))]
    plt.plot(set2d[:, 0], set2d[:, 1], 'bo', markersize=16)
    plt.plot(pfx, pfy, 'ro', markersize=16)
    plt.fill(pfx, pfy, facecolor='red', alpha=0.3)
    plt.xlabel('Ziel 1')
    plt.ylabel('Ziel 2')
    plt.title('')
    plt.grid(False)
    plt.show()