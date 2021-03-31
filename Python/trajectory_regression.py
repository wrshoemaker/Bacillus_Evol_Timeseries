from __future__ import division
import os, sys, json, pickle
import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import statsmodels.stats.multitest as multitest

import phylo_tools as pt

import scipy.stats as stats

import parse_file
import timecourse_utils
import mutation_spectrum_utils



with open(pt.get_path() + '/data/mutation_dynamics.pickle', 'rb') as handle:
    r2s_obs_dict = pickle.load(handle)


#analyses = ['f_max', 'abs_delta_f', 'ratio_f']

fig, ax = plt.subplots(figsize=(4,4))

analysis = 'abs_delta_f'

for taxon in pt.taxa:

    for treatment_idx, treatment in enumerate(pt.treatments):


        y = r2s_obs_dict[treatment][taxon][analysis]

        x = [pt.B_S_generation_dict[taxon][treatment]]* len(y)

        ax.scatter(x, y, alpha=0.4, zorder=2)


ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.set_ylim([0.0000i001, 0.1])

fig.subplots_adjust(hspace=0.3, wspace=0.5)
fig_name = pt.get_path() + '/figs/trajectory_regression.jpg'
fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
