from __future__ import division
import os, sys, json, pickle
import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import statsmodels.stats.multitest as multitest

import phylo_tools as pt

import scipy.stats as stats

import parse_file
import timecourse_utils
import mutation_spectrum_utils



#import matplotlib
cmap = 'YlOrRd'
c = 0.00001
N = 10**6

np.random.seed(123456789)


label_latex_dict = {'abs_delta_f': r'$\frac{\left | \Delta f \right |}{\Delta \tau}$',
                    'ratio_f': r'$\frac{f(t+ \delta t)}{f(t)}$',
                    'r2': r'$\rho^{2}_{M^{(i)}, M^{(j)} } $',
                    'f_max': r'$f_{max}$'}


label_text_dict = {'abs_delta_f': 'Absolute change in allele frequencies\nper-generation',
                    'ratio_f': 'Ratio of allele frequency changes',
                    'r2': 'Squared correlation between\nallele frequency trajectories',
                    'f_max': 'Maximum observed allele frequency'}




fig = plt.figure(figsize = (9, 8))
gs = gridspec.GridSpec(nrows=3, ncols=2)

row_count = 0

ax_sim_f_max = fig.add_subplot(gs[0, 0])
ax_sim_delta_f = fig.add_subplot(gs[1, 0])
ax_sim_ratio_f = fig.add_subplot(gs[2, 0])

ax_heatmap_f_max = fig.add_subplot(gs[0, 1])
ax_heatmap_delta_f = fig.add_subplot(gs[1, 1])
ax_heatmap_ratio_f = fig.add_subplot(gs[2, 1])

xlim_dict = {'abs_delta_f': [10**-6, 10**-2],
            'ratio_f': [3, 3000],
            'f_max': [ 10**-6, 10**-2 ]}


ylim_dict = {'abs_delta_f': [0.0005, 1.1],
            'ratio_f': [0.0005, 1.1],
            'f_max': [ 0.0008, 1.03 ]}


# plot simulations
saved_data_file='%s/data/simulations/all_seedbank_sizes.dat' % (pt.get_path())
sampled_timepoints = pickle.load( open(saved_data_file, "rb" ) )
metrics = ['max_f', 'delta_f', 'ratio_f']
ax_sim_list = [ax_sim_f_max, ax_sim_delta_f, ax_sim_ratio_f]
m_for_sims = [10, 4281, 297635]
sim_treatment_list = ['0', '1', '2']
for metric_idx, metric in enumerate(metrics):

    ax_sim = ax_sim_list[metric_idx]
    ax_sim.set_yscale('log', base=10)
    ax_sim.set_xscale('log', base=10)

    ax_sim.xaxis.set_tick_params(labelsize=8)
    ax_sim.yaxis.set_tick_params(labelsize=8)

    if metric == 'max_f':
        analysis = 'f_max'
    elif metric == 'delta_f':
        analysis = 'abs_delta_f'
    else:
        analysis = 'ratio_f'


    ax_sim.set_xlabel('%s, %s' % (label_text_dict[analysis], label_latex_dict[analysis] ) , fontsize = 11)
    ax_sim.set_yscale('log', base=10)
    ax_sim.set_xlim(xlim_dict[analysis])

    ax_sim.set_ylim(ylim_dict[analysis])
    #if count != 2:
    ax_sim.set_xscale('log', base=10)

    ax_sim.set_ylabel('Fraction ' + r'$\geq$' + label_latex_dict[analysis], fontsize=11)


    for M_idx, M in enumerate(m_for_sims):
        f_max_array_sort = np.sort(sampled_timepoints[M][c][metric])
        cdf = 1-np.arange(len(f_max_array_sort))/float(len(f_max_array_sort))
        sim_treatment = sim_treatment_list[M_idx]

        time_seedbank = M/(c*N)

        ax_sim.plot(f_max_array_sort, cdf, c =pt.get_colors(sim_treatment), ls='-', lw=3, alpha=0.8, label= r'$\left \langle T_{d} \right \rangle=$' + str(int(time_seedbank)))

    if (metric_idx == 0) or (metric_idx == 1):
        ax_sim.legend(loc='lower left', fontsize=6)

    else:
        ax_sim.legend(loc='upper right', fontsize=6)




# heatmaps

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



metric_range_dict = {'max_f':(-7, -1), 'delta_f': (-7,-1), 'ratio_f':(-4,4)}
metric_x_axis_range_dict = {'max_f':(-3, -1), 'delta_f': (-5,-1), 'ratio_f':(0.1,2.3)}


ax_heatmap_list = [ax_heatmap_f_max, ax_heatmap_delta_f, ax_heatmap_ratio_f ]





M_list = list( sampled_timepoints.keys())
M_list.sort()
M_list = np.asarray(M_list)
for metric_idx, metric in enumerate(metrics):

    bins_ = np.logspace(metric_range_dict[metric][0], metric_range_dict[metric][1], base=10.0, num=51)
    bins_x_axis = bins_[1:]
    bins_x_axis = np.log10(bins_x_axis)
    x_axis = np.vstack([bins_x_axis]*len(M_list))
    y_axis = np.vstack([M_list]*len(bins_x_axis))
    y_axis = np.log10(y_axis)

    all_cdf = []

    for M in M_list:

        K = N/M

        t_dormant = 1/(c*K)

        dict_M = sampled_timepoints[M][c]

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



    #fig, ax = plt.subplots(figsize=(4,2))

    ax = ax_heatmap_list[metric_idx]

    #im, cbar = heatmap(all_cdf, vegetables, M_list, ax=ax,
    #                   cmap="YlGn", cbarlabel="harvest [t/year]")


    pcm = ax.pcolor(x_axis, y_axis, all_cdf,
                       norm=colors.LogNorm(vmin=all_cdf.min(), vmax=all_cdf.max()),
                       cmap='YlOrRd', shading='auto')

    ax.set_xlim(metric_x_axis_range_dict[metric])

    ax.xaxis.set_major_formatter(fake_log_float)
    ax.yaxis.set_major_formatter(fake_log)


    if metric == 'max_f':
        analysis = 'f_max'
    elif metric == 'delta_f':
        analysis = 'abs_delta_f'
    else:
        analysis = 'ratio_f'


    ax.set_xlabel('%s, %s' % (label_text_dict[analysis], label_latex_dict[analysis] ) , fontsize = 11)


    clb = plt.colorbar(pcm, ax=ax)
    #clb.ax.set_title(r'$T_{\mathrm{seed \,bank}}$')

    clb.set_label(label='Fraction ' + r'$\geq$' + label_latex_dict[analysis])

    #ax_sim.set_ylabel('Fraction ' + r'$\geq$' + label_latex_dict[analysis], fontsize=11)


    ax.set_ylabel('Avg. time in seedbank, ' + r'$\left \langle T_{d} \right \rangle$', fontsize=8)

    #analysis_ax.set_ylabel('Fraction ' + r'$\geq$' + label_latex_dict[analysis], fontsize=11)








ax_list = [ax_sim_f_max, ax_sim_delta_f, ax_sim_ratio_f, ax_heatmap_f_max, ax_heatmap_delta_f, ax_heatmap_ratio_f]

for ax_idx, ax in enumerate(ax_list):
    ax.text(-0.1, 1.07, pt.sub_plot_labels[ax_idx], fontsize=8, fontweight='bold', ha='center', va='center', transform=ax.transAxes)












fig_name = pt.get_path() + '/figs/simulation_CDF.jpg'
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
