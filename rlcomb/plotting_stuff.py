"""
Created on Feb 10, 2015

@author: Dominik Meyer <meyerd@mytum.de>
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_that_pretty_rldm15(xdata=[], ydata=[], labels=[],
                            xlabel="", x_range=(), ylabel="", y_range=(),
                            output_filename="", custom_yticks=None,
                            fontsize=16, label_fontsize=16,
                            label_offsets=None, y_lim=None, x_lim=None):
        # from matplotlib import rc
        # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        # rc('text', usetex=True)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        for i in range(len(tableau20)):
            r, g, b = tableau20[i]
            tableau20[i] = (r / 255., g / 255., b / 255.)

        color_permutation = [2, 4, 0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                             15, 16, 17, 18, 19]

        # Common sizes: (10, 7.5) and (12, 9)
        # figure(figsize=(12, 14))
        plt.figure(figsize=(10, 7.5))
        # plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        if y_lim:
            plt.ylim(y_lim)
        if x_lim:
            plt.xlim(x_lim)

        if custom_yticks:
            plt.yticks(np.arange(*y_range), custom_yticks,
                       fontsize=fontsize)
        else:
            plt.yticks(np.arange(*y_range), ["%.1f" % (x) for x in
                                             np.arange(*y_range)],
                       fontsize=fontsize)
        plt.xticks(np.arange(*x_range), ["%.1f" % (x) for x in
                                         np.arange(*x_range)],
                   fontsize=fontsize)

        mod_y_range = (y_range[0] + y_range[2], y_range[1], y_range[2])
        for y in np.arange(*mod_y_range):
            plt.plot(np.arange(*x_range), [y] * len(np.arange(*x_range)), "--",
                     lw=0.5, color="black", alpha=0.3)

        plt.tick_params(axis="both", which="both", bottom="on", top="off",
                        labelbottom="on", left="off", right="off",
                        labelleft="on")

        plt.xlabel(xlabel, fontsize=label_fontsize, alpha=0.7)
        plt.ylabel(ylabel, fontsize=label_fontsize, alpha=0.7)

        linelabels = []
        if not label_offsets:
            label_offsets = []
            for _ in xdata:
                label_offsets.append(0.0)

        for dp, _ in enumerate(xdata):
            plt.plot(xdata[dp], ydata[dp], label=labels[dp],
                     lw=2.5, color=tableau20[color_permutation[dp]])
#             plt.fill_between(np.linspace(linspace_from, linspace_to,
#                                          linspace_steps),
#                              avg_payouts1 + std_payouts1,
#                              vg_payouts1 - std_payouts1,
#                              facecolor=tableau20[2], alpha=0.3)

            # Again, make sure that all labels are large enough to be easily
            # read by the viewer.
            y_pos = ydata[dp][-1] - 0.5
            tx = plt.text(1.01, y_pos+.5+label_offsets[dp], labels[dp],
                          fontsize=fontsize,
                          color=tableau20[color_permutation[dp]])
            linelabels.append(tx)

        # plt.legend(loc='upper left')
        # plt.savefig(output_filename)
        plt.savefig(output_filename, bbox_extra_artists=linelabels,
                    bbox_inches='tight')
