from __future__ import division
import os, sys, json, pickle
import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt

import matplotlib

from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.colors as colors
import matplotlib.cbook as cbook


import statsmodels.stats.multitest as multitest

import phylo_tools as pt

import scipy.stats as stats


import parse_file
import timecourse_utils
import mutation_spectrum_utils

# avg time in dormant state  = c*K*M / M =

cmap = 'YlOrRd'

c = 1e-05
N = 10**6

with open(pt.get_path() + '/data/simulations/all_seedbank_sizes.dat', 'rb') as handle:
    simulation_data = pickle.load(handle)


#metrics = ['delta_f', 'ratio_f', 'max_f']
metrics = ['ratio_f']

metric_range_dict = {'max_f':(-7, -1), 'delta_f': (-7,-1), 'ratio_f':(-4,4)}


metric_x_axis_range_dict = {'max_f':(-1, 0), 'delta_f': (-6,-2), 'ratio_f':(-0.8,2.3)}



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar




@plt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%d}$' % (  x )


@plt.FuncFormatter
def fake_log_float(x, pos):
    'The two args are the value and tick position'

    return r'$10^{%.1f}$' % (  x )



M_list = list( simulation_data.keys())
M_list.sort()
M_list = np.asarray(M_list)
for metric in metrics:

    bins_ = np.logspace(metric_range_dict[metric][0], metric_range_dict[metric][1], base=10.0, num=51)
    print(bins_)

    bins_x_axis = bins_[1:]
    bins_x_axis = np.log10(bins_x_axis)
    x_axis = np.vstack([bins_x_axis]*len(M_list))
    y_axis = np.vstack([M_list]*len(bins_x_axis))

    #print(y_axis)
    y_axis = np.log10(y_axis)

    print(y_axis)


    all_cdf = []

    for M in M_list:

        K = N/M

        t_dormant = 1/(c*K)

        dict_M = simulation_data[M][c]

        array_sort = np.sort(dict_M[metric])
        array_sort = array_sort[array_sort>0]
        array_sort = np.log10(array_sort)

        cdf_bins = []
        for bins_dx in range(len(bins_)-1):

            bin_lower = np.log10(bins_[bins_dx])
            bin_upper = np.log10(bins_[bins_dx+1])

            n_bin = len(array_sort[(array_sort > bin_lower) & (array_sort <= bin_upper)])

            cdf_bins.append((n_bin+1) / (len(array_sort) + 50))

        all_cdf.append(cdf_bins)



    all_cdf = np.asarray(all_cdf)
    y_axis = np.transpose(y_axis)



    fig, ax = plt.subplots(figsize=(4,2))

    #im, cbar = heatmap(all_cdf, vegetables, M_list, ax=ax,
    #                   cmap="YlGn", cbarlabel="harvest [t/year]")


    pcm = ax.pcolor(x_axis, y_axis, all_cdf,
                       norm=colors.LogNorm(vmin=all_cdf.min(), vmax=all_cdf.max()),
                       cmap='YlOrRd', shading='auto')

    ax.set_xlim(metric_x_axis_range_dict[metric])

    ax.xaxis.set_major_formatter(fake_log_float)
    ax.yaxis.set_major_formatter(fake_log)

    clb = plt.colorbar(pcm, ax=ax)
    #clb.ax.set_title(r'$T_{\mathrm{seed \,bank}}$')

    clb.set_label(label='Fraction ' + r'$\geq$')


    ax.set_ylabel('Avg. time in seedbank, ' + r'$T_{\mathrm{seed \,bank}}$', fontsize=8)

    #analysis_ax.set_ylabel('Fraction ' + r'$\geq$' + label_latex_dict[analysis], fontsize=11)



    fig_name = pt.get_path() + '/figs/simulation_heatmap.jpg'
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



#norm=colors.LogNorm(vmin=Z.min()
