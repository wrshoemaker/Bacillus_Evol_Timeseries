from __future__ import division
import os, sys
import numpy as np

import phylo_tools as pt
import parse_file
import timecourse_utils

import  matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.gridspec as gridspec


populations = [['0B3', '1B4', '1B5'], ['0B5', '1B5', '2B2']]


fig = plt.figure(figsize = (12, 6))
gs = gridspec.GridSpec(nrows=2, ncols=3)

row_count = 0


for row_i_idx, row_i  in enumerate(populations):

    for column_j_idx, column_j in enumerate(row_i):

        annotated_timecourse_path = pt.get_path() + "/data/timecourse_final/%s_annotated_timecourse.txt" % column_j
        if os.path.exists(annotated_timecourse_path) == False:
            continue
        annotated_timecourse_file = open(annotated_timecourse_path ,"r")


        ax_i_j = fig.add_subplot(gs[ row_i_idx, column_j_idx])

        first_line = annotated_timecourse_file.readline()
        first_line = first_line.strip()
        first_line_items = first_line.split(",")
        times = np.asarray([float(x.strip().split(':')[1]) for x in first_line_items[16::2]])
        times = np.insert(times, 0, 0, axis=0)


        times_to_ignore = None


        if times_to_ignore != None:
            times_to_ignore_idx=[np.where(times == t)[0] for t in times_to_ignore][0]
            times = np.delete(times, times_to_ignore_idx)

        for i, line in enumerate(annotated_timecourse_file):
            line = line.strip()
            items = line.split(",")
            pass_or_fail = items[15].strip()
            if pass_or_fail == 'FAIL':
                continue

            alt_cov = np.asarray([ float(x) for x in items[16::2]])
            total_cov = np.asarray([ float(x) for x in items[17::2]])
            # pseudocount to avoid divide by zero error
            alt_cov = alt_cov
            total_cov = total_cov + 1
            freqs = alt_cov / total_cov
            freqs = np.insert(freqs, 0, 0, axis=0)

            if times_to_ignore != None:
                freqs = np.delete(freqs, times_to_ignore_idx)

            rgb = pt.mut_freq_colormap()
            rgb = pt.lighten_color(rgb, amount=0.5)

            if len(times) == len(freqs) + 1:
                freqs = np.insert(freqs, 0, 0, axis=0)

            ax_i_j.plot(times, freqs, '.-', c=rgb, alpha=0.4)

        ax_i_j.set_xlim([0,700])
        ax_i_j.set_ylim([0, 1])

        #if column_count == 0:
        #    ax_i_j.set_ylabel( str(10**int(treatment)) + '-day transfers', fontsize =12  )

        ax_i_j.tick_params(axis="x", labelsize=8)
        ax_i_j.tick_params(axis="y", labelsize=8)

        if row_i_idx == 1:
            ax_i_j.set_xlabel('Days, ' + r'$t$', fontsize = 8)

        if column_j_idx == 0:
            ax_i_j.set_ylabel('Allele frequency, ' + r'$f(t)$', fontsize=8)




#fig.text(0.5, 0.04, 'Days, ' + r'$t$', ha='center', va='center', fontsize=24)
#fig.text(0.05, 0.5, 'Allele frequency, ' + r'$f(t)$', ha='center', va='center', rotation='vertical',  fontsize=24)

#fig.suptitle(pt.latex_dict[strain], fontsize=28, fontweight='bold')
fig_name = '%s/figs/example_trajectories.jpg' % pt.get_path()
fig.savefig(fig_name, bbox_inches = "tight",  format='jpg',  pad_inches = 0.4)#, dpi = 600)

plt.close()
