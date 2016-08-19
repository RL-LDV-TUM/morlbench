import numpy as np
import matplotlib.pyplot as plt
import random
from morlbench.helpers import HyperVolumeCalculator
if __name__ == '__main__':
    u = 20
    ref_point2d = [0.1, 0.1]
    set2d = np.zeros((u, 2))
    for i in range(u):
        for u in range(2):
            rand = random.random()
            set2d[i, u] = rand if (rand > ref_point2d) else random.random()
    hv_2d_calc = HyperVolumeCalculator(ref_point2d)
    pf = hv_2d_calc.extract_front(set2d)
    plt.figure()
    plt.axis([0-0.1, max(set2d[:, 0]*1.21), 0-0.1, max(set2d[:, 1]*1.1)])
    pfx = [pf[i][0] for i in range(len(pf))]
    pfy = [pf[u][1] for u in range(len(pf))]
    plt.plot(set2d[:, 0], set2d[:, 1], 'ro', markersize=15)
    plt.plot(pfx, pfy, 'bo', markersize=15)
    # plt.plot(ref_point2d[0], ref_point2d[1], 'mo', markersize=20)
    # plt.fill_betweenx(pfx, 0, pfy, facecolor='blue', alpha=0.5)
    plt.xlabel('Ziel 1')
    plt.ylabel('Ziel 2')
    plt.title('')
    plt.grid(False)
    plt.show()