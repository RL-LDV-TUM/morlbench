import matplotlib.pyplot as plt

if __name__ == '__main__':

    a = [1, 1, 4]
    b = [1, 3, 2.5]
    fig = plt.figure()
    plt.grid()

    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    plt.plot(a, b, 'ro', markersize=20, zorder=1)
    u = 0
    for i, j in zip(a, b):
        ax.annotate('$P_%i$' % u, xy=(i, j), xytext=(0, 20), size=20, textcoords='offset points')
        ax.annotate('('+str(i)+','+str(j)+')', xy=(i, j), xytext=(25, 23), size=10, textcoords='offset points')
        u+=1

    ax.annotate("dominiert strikt", weight="bold",
            xy=(1.1, 0.88), xycoords='data',
            xytext=(2.7, 1.2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.3"),
            bbox=dict(boxstyle="square", fc="w")
            )
    ax.annotate("",
            xy=(3.33, 1.36), xycoords='data',
            xytext=(4.0, 2.3), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3,rad=-0.17"),
            )

    ax.annotate("dominiert schwach", weight="bold",
            xy=(0.83, 1.1), xycoords='data',
            xytext=(-0.3, 2.1), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0.3"),
            bbox=dict(boxstyle="square", fc="w")
            )
    ax.annotate("",
            xy=(0.42, 2.29), xycoords='data',
            xytext=(0.9, 2.88), textcoords='data',
            arrowprops=dict(arrowstyle="-",
                            connectionstyle="arc3,rad=0.22"),
            bbox=dict(boxstyle="square", fc="w")
            )

    ax.annotate("nicht vergleichbar", weight="bold",
            xy=(3.9, 2.68), xycoords='data',
            xytext=(2.1, 3.1), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.3"),
            bbox=dict(boxstyle="square", fc="w")
            )
    ax.annotate("",
            xy=(2.0, 3.2), xycoords='data',
            xytext=(1.14, 3.02), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3,rad=-0.1"),
            )

    plt.xlabel('Ziel 1')
    plt.ylabel('Ziel 2')
    plt.axis([-0.5, 4.7, 0, 4])
    plt.show()