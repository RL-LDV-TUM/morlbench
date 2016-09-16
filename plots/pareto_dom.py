
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mpl.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    a = [1.0, 1.0, 4.0]
    b = [1.0, 3.0, 2.5]
    size = 0.48 * 5.8091048611149611602
    fig = plt.figure(figsize=[size, 0.75 * size])

    fig.set_size_inches(size, 0.7 * size)

    ax = fig.add_subplot(111)
    plt.grid(zorder=0)

    #ax.set_axisbelow(True)
    plt.plot(a, b, 'ro', markersize=4, zorder=3)

    ax.annotate('$P_0$', xy=(a[0], b[0]), xytext=(a[0]-0.5, b[0]-0.55), size=11)
    ax.annotate('('+str(a[0])+','+str(b[0])+')', xy=(a[0], b[0]), xytext=(a[0]+0.1, b[0]+0.1), size=6)
    ax.annotate('$P_1$', xy=(a[1], b[1]), xytext=(a[1] - 0.47, b[1]+0.17), size=11)
    ax.annotate('(' + str(a[1]) + ',' + str(b[1]) + ')', xy=(a[1], b[1]), xytext=(a[1] + 0.1, b[1] - 0.35), size=6)
    ax.annotate('$P_2$', xy=(a[2], b[2]), xytext=(a[2]+0.1, b[2]), size=11)
    ax.annotate('(' + str(a[2]) + ',' + str(b[2]) + ')', xy=(a[2], b[2]), xytext=(a[2] - 0.9, b[2] - 0.2), size=6)

    ax.annotate("dominiert\n  strikt", size=9,
            xy=(1.1, 0.9), xycoords='data',
            xytext=(2.5, 0.7), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.3"),
            bbox=dict(boxstyle="square", fc="w")
            )
    ax.annotate("",
            xy=(3.6, 1.60), xycoords='data',
            xytext=(4.0, 2.3), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3,rad=-0.17"),
            )

    ax.annotate("dominiert\n  schwach", size=9,
            xy=(1.01, 1.02), xycoords='data',
            xytext=(-0.1, 1.7), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0.3"),
            bbox=dict(boxstyle="square", fc="w")
            )
    ax.annotate("",
            xy=(0.6, 2.5), xycoords='data',
            xytext=(1.0, 2.99), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3,rad=0.22"),
            bbox=dict(boxstyle="square", fc="w")
            )

    ax.annotate("   nicht\nvergleichbar", size=9,
            xy=(4.0, 2.55), xycoords='data',
            xytext=(1.9, 3.3), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.3"),
            bbox=dict(boxstyle="square", fc="w")
            )
    ax.annotate("",
            xy=(1.80, 3.2), xycoords='data',
            xytext=(1.14, 3.02), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3,rad=-0.1"),
            )

    plt.xlabel('Ziel 1', size=9)
    plt.ylabel('Ziel 2', size=9)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    plt.axis([-0.5, 4.7, 0.1, 4.35])
    plt.xticks(np.arange(0.0, 5.0, 1.0))
    plt.yticks(np.arange(0.0, 5.0, 1.0))
    plt.subplots_adjust(bottom=0.20, left=0.17)

    plt.show()
