import numpy as np
import matplotlib.pyplot as plt
import random
from morlbench.helpers import HyperVolumeCalculator
import matplotlib as mpl
if __name__ == '__main__':
    random.seed(3323)
    mpl.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    u = 20
    ref_point2d = [0.1, 0.1]
    set2d = np.zeros((u, 2))
    for i in range(u):
        for u in range(2):
            rand = random.random()
            set2d[i, u] = rand if (rand > ref_point2d) else random.random()
    hv_2d_calc = HyperVolumeCalculator(ref_point2d)
    pf = hv_2d_calc.extract_front(set2d)
    size = 0.48 * 5.8091048611149611602
    fig = plt.figure(figsize=[size, 0.75 * size])
    fig.set_size_inches(size, 0.7 * size)
    ax = fig.add_subplot(1,1,1)
    plt.axis([0-0.1, max(set2d[:, 0]*1.21), 0-0.1, max(set2d[:, 1]*1.1)])
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    pfx = [pf[i][0] for i in range(len(pf))]
    pfy = [pf[u][1] for u in range(len(pf))]
    plt.plot(set2d[:, 0], set2d[:, 1], 'ro', markersize=4)
    plt.plot(pfx, pfy, 'bo', markersize=4)
    # plt.plot(ref_point2d[0], ref_point2d[1], 'mo', markersize=20)
    # plt.fill_betweenx(pfx, 0, pfy, facecolor='blue', alpha=0.5)
    plt.xlabel('Ziel 1', size=9)
    plt.ylabel('Ziel 2', size=9)
    plt.title('')
    plt.subplots_adjust(bottom=0.20, left=0.17)
    plt.grid(False)
    plt.show()