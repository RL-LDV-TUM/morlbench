import numpy as np
import matplotlib.pyplot as plt
import random
from morlbench.helpers import HyperVolumeCalculator
from matplotlib.font_manager import FontProperties
from inspyred.ec.analysis import hypervolume
if __name__ == '__main__':

    count = 20
    random.seed(18.9654)
    ref_point2d = [0.0002, 0.0002]
    set2d = np.zeros((count, 2))
    for i in range(count):
        for u in range(2):
            rand = random.random()
            set2d[i, u] = rand if (rand > ref_point2d)or(rand > 0.3) else random.random()
    hv_2d_calc = HyperVolumeCalculator(ref_point2d)
    pf = hv_2d_calc.extract_front(set2d)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    fm = FontProperties()
    fm.set_family('serif')
    # make red surfaces
    right_bottomx = [1.0, 1.21, 1.21, 1.0, 1.0]
    right_bottomy = [0.0, 0.0, -0.12, -0.12, 0.0]
    plt.plot(right_bottomx, right_bottomy, 'r--')
    plt.fill_betweenx(right_bottomy, right_bottomx, facecolor='red', alpha=0.2)
    left_topx = [0.0, 0.0, -0.3, -0.3, 0.0]
    left_topy = [0.9, 1.4, 1.4, 0.88, 0.88]
    plt.plot(left_topx, left_topy, 'r--')
    plt.fill_between(left_topx, left_topy, facecolor='red', alpha=0.2)
    ###########################################################################################
    plt.axis([-0.16, max(set2d[:, 0]*1.16), 0-0.1, max(set2d[:, 1]*1.1)], fontproperties='serif')
    pfx = [pf[i][0] for i in range(len(pf))]
    pfy = [pf[u][1] for u in range(len(pf))]
    maxx = [ref_point2d[0], max(pfx)]
    maxx.extend(pfx)
    pfx = maxx
    miny = [ref_point2d[0], ref_point2d[1]]
    miny.extend(pfy)
    pfy = miny
    minx = ref_point2d[0]
    pfx.extend([minx, ref_point2d[0]])
    pfy.extend([max(pfy), ref_point2d[1]])
    plt.plot(set2d[:, 0], set2d[:, 1], 'ro', markersize=10)
    plt.plot(pfx, pfy, 'bo', linestyle='--', drawstyle='steps-post', markersize=10)
    plt.plot(ref_point2d[0], ref_point2d[1], 'ko', markersize=20)
    xy = (ref_point2d[0], ref_point2d[1])
    ax.annotate("Reference Point", xytext=(0.01, -0.06), xy=xy, fontproperties='serif')
    ax.annotate("tl", xytext=(-0.05, 0.9), xy=(-0.05, 0.9), fontproperties='serif')
    ax.annotate("br", xytext=(1.02, -0.05), xy=(1.03, -0.05), fontproperties='serif')
    new_pfx = pfx[:1]
    new_pfy = pfy[:1]
    for i in xrange(1, len(pfx)-2):
        new_pfx.append(pfx[i])
        new_pfx.append(pfx[i+1])
    for u in xrange(1, len(pfy)-2):
        new_pfy.append(pfy[u])
        new_pfy.append(pfy[u])
    new_pfx.extend(pfx[len(pfx)-1:])
    new_pfy.extend(pfy[len(pfy)-1:])

    plt.fill_betweenx(new_pfy, new_pfx, facecolor='blue', alpha=0.2)
    plt.xlabel('Ziel 1', fontproperties='serif')
    plt.ylabel('Ziel 2', fontproperties='serif')
    plt.title('')
    plt.grid(False)
    plt.show()