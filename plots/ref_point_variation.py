import numpy as np
import matplotlib.pyplot as plt
import random
from morlbench.helpers import HyperVolumeCalculator
from matplotlib.font_manager import FontProperties
from inspyred.ec.analysis import hypervolume
import matplotlib as mpl
if __name__ == '__main__':
    mpl.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    count = 20
    random.seed(18.9654)
    ref_point2d = [0.0, 0.3]
    set2d = np.zeros((count, 2))
    for i in range(count):
        for u in range(2):
            rand = random.random()
            set2d[i, u] = rand if (rand > ref_point2d)or(rand > 0.3) else random.random()
    hv_2d_calc = HyperVolumeCalculator(ref_point2d)
    pf = hv_2d_calc.extract_front(set2d)
    hv = hv_2d_calc.compute_hv(pf)
    size = 0.48*5.8091048611149611602
    fig = plt.figure(figsize=[size, 0.75*size])

    fig.set_size_inches(size, 0.7*size)

    ax = fig.add_subplot(111)

    ax.set_axisbelow(True)


    ###########################################################################################
    plt.axis([-0.06, max(set2d[:, 0]*1.06), 0-0.18, max(set2d[:, 1]*1.07)])
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    pfx = [pf[i][0] for i in range(len(pf))]
    pfy = [pf[u][1] for u in range(len(pf))]
    maxx = [ref_point2d[0], max(pfx)]
    maxx.extend(pfx)
    pfx = maxx
    miny = [ref_point2d[1], ref_point2d[1]]
    miny.extend(pfy)
    pfy = miny
    minx = ref_point2d[0]
    pfx.extend([minx, ref_point2d[0]])
    pfy.extend([max(pfy), ref_point2d[1]])
    plt.plot(set2d[:, 0], set2d[:, 1], 'ro', markersize=4)
    plt.plot(pfx, pfy, 'bo', linestyle='--', drawstyle='steps-post', markersize=4)
    plt.plot(ref_point2d[0], ref_point2d[1], 'ko', markersize=10)
    xy = (ref_point2d[0], ref_point2d[1])
    ax.annotate("Reference Point="+str(ref_point2d)+"$^T$", xytext=(ref_point2d[0]+0.04, ref_point2d[1]-0.13), xy=xy,
                fontsize=9)
    ax.annotate("Hypervolume: " + str(round(hv, 2)), (0.10, 0.44), fontsize=9, color='blue')
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
    plt.xlabel('Ziel 1', size=9)
    plt.ylabel('Ziel 2', size=9)
    plt.title('')
    plt.grid(False)
    plt.subplots_adjust(bottom=0.19, left=0.18)
    plt.show()

    count = 20
    random.seed(18.9654)
    ref_point2d = [0.0, 0.0]
    set2d = np.zeros((count, 2))
    for i in range(count):
        for u in range(2):
            rand = random.random()
            set2d[i, u] = rand if (rand > ref_point2d) or (rand > 0.3) else random.random()
    hv_2d_calc = HyperVolumeCalculator(ref_point2d)
    pf = hv_2d_calc.extract_front(set2d)
    hv = hv_2d_calc.compute_hv(pf)
    size = 0.48 * 5.8091048611149611602
    fig = plt.figure(figsize=[size, 0.75 * size])

    fig.set_size_inches(size, 0.7 * size)

    ax = fig.add_subplot(111)


    ###########################################################################################
    plt.axis([-0.06, max(set2d[:, 0] * 1.06), 0 - 0.18, max(set2d[:, 1] * 1.07)])
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    pfx = [pf[i][0] for i in range(len(pf))]
    pfy = [pf[u][1] for u in range(len(pf))]
    maxx = [ref_point2d[0], max(pfx)]
    maxx.extend(pfx)
    pfx = maxx
    miny = [ref_point2d[1], ref_point2d[1]]
    miny.extend(pfy)
    pfy = miny
    minx = ref_point2d[0]
    pfx.extend([minx, ref_point2d[0]])
    pfy.extend([max(pfy), ref_point2d[1]])
    plt.plot(set2d[:, 0], set2d[:, 1], 'ro', markersize=4)
    plt.plot(pfx, pfy, 'bo', linestyle='--', drawstyle='steps-post', markersize=4)
    plt.plot(ref_point2d[0], ref_point2d[1], 'ko', markersize=10)
    xy = (ref_point2d[0], ref_point2d[1])
    ax.annotate("Reference Point=" + str(ref_point2d) + "$^T$", xytext=(ref_point2d[0] + 0.04, ref_point2d[1] - 0.12),
                xy=xy,
                fontsize=9)
    ax.annotate("Hypervolume: " + str(round(hv, 2)), (0.1, 0.44), fontsize=9, color='blue')

    new_pfx = pfx[:1]
    new_pfy = pfy[:1]
    for i in xrange(1, len(pfx) - 2):
        new_pfx.append(pfx[i])
        new_pfx.append(pfx[i + 1])
    for u in xrange(1, len(pfy) - 2):
        new_pfy.append(pfy[u])
        new_pfy.append(pfy[u])
    new_pfx.extend(pfx[len(pfx) - 1:])
    new_pfy.extend(pfy[len(pfy) - 1:])

    plt.fill_betweenx(new_pfy, new_pfx, facecolor='blue', alpha=0.2)
    plt.xlabel('Ziel 1', size=9)
    plt.ylabel('Ziel 2', size=9)
    plt.title('')
    plt.grid(False)
    plt.subplots_adjust(bottom=0.19, left=0.18)
    plt.show()