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


np.random.seed(123456789)

treatments=pt.treatments
replicates = pt.replicates

min_trajectory_length=3


taxa = ['B', 'S']

def run_analyses():
    r2s_obs_dict = {}
    #r2s_null_dict = {}
    for treatment in ['0', '1', '2']:
        r2s_obs_dict[treatment] = {}
        for taxon in taxa:
            r2s_all = []
            ratio_f_all = []
            abs_delta_f_all = []
            for replicate in replicates:

                population = treatment + taxon + replicate
                sys.stderr.write("Processing %s...\n" % population)

                mutations, depth_tuple = parse_file.parse_annotated_timecourse(population)
                population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths = depth_tuple
                state_times, state_trajectories = parse_file.parse_well_mixed_state_timecourse(population)

                times = mutations[0][12]
                Ms = np.zeros_like(times)*1.0
                fixed_Ms = np.zeros_like(times)*1.0

                for mutation_idx_i in range(0,len(mutations)):

                    location_i, gene_name_i, allele_i, var_type_i, codon_i, position_in_codon_i, AAs_count_i, test_statistic_i, pvalue_i, cutoff_idx_i, depth_fold_change_i, depth_change_pvalue_i, times_i, alts_i, depths_i, clone_times_i, clone_alts_i, clone_depths_i = mutations[mutation_idx_i]

                    state_Ls_i = state_trajectories[mutation_idx_i]
                    good_idx_i, filtered_alts_i, filtered_depths_i = timecourse_utils.mask_timepoints(times_i, alts_i, depths_i, var_type_i, cutoff_idx_i, depth_fold_change_i, depth_change_pvalue_i)
                    freqs_i = timecourse_utils.estimate_frequencies(filtered_alts_i, filtered_depths_i)

                    masked_times_i = times[good_idx_i]
                    masked_freqs_i = freqs_i[good_idx_i]
                    masked_state_Ls_i = state_Ls_i[good_idx_i]

                    P_idx_i = np.where(masked_state_Ls_i == 3)[0]
                    if len(P_idx_i) < min_trajectory_length:
                        continue
                    first_P_i = P_idx_i[0]
                    last_P_i = P_idx_i[-1]

                    masked_freqs_P_i = masked_freqs_i[first_P_i:last_P_i+1]
                    masked_times_P_i = masked_times_i[first_P_i:last_P_i+1]

                    delta_masked_freqs_P_i = masked_freqs_P_i[1:] - masked_freqs_P_i[:-1]
                    delta_masked_times_P_i = masked_times_P_i[:-1]

                    #abs_delta_f = np.absolute(freqs_i[1:] - freqs_i[:-1])
                    #freqs_i_no_zero = freqs_i[freqs_i>0]
                    # we want to get the ratio of freqs

                    for freqs_i_k, freqs_i_l in  zip(freqs_i[1:], freqs_i[:-1]):
                        if (freqs_i_k == 0) or (freqs_i_l == 0):
                            continue
                        abs_delta_f_all.append(np.absolute(freqs_i_k-freqs_i_l))
                        ratio_f_all.append(freqs_i_k/freqs_i_l)


                    #ratio_f = freqs_i_no_zero[]


                    for mutation_idx_j in range(mutation_idx_i+1,len(mutations)):

                        location_j, gene_name_j, allele_j, var_type_j, codon_j, position_in_codon_j, AAs_count_j, test_statistic_j, pvalue_j, cutoff_jdx_j, depth_fold_change_j, depth_change_pvalue_j, times_j, alts_j, depths_j, clone_times_j, clone_alts_j, clone_depths_j = mutations[mutation_idx_j]

                        state_Ls_j = state_trajectories[mutation_idx_j]
                        good_idx_j, filtered_alts_j, filtered_depths_j = timecourse_utils.mask_timepoints(times_j, alts_j, depths_j, var_type_j, cutoff_jdx_j, depth_fold_change_j, depth_change_pvalue_j)
                        freqs_j = timecourse_utils.estimate_frequencies(filtered_alts_j, filtered_depths_j)

                        masked_times_j = times[good_idx_j]
                        masked_freqs_j = freqs_j[good_idx_j]
                        masked_state_Ls_j = state_Ls_j[good_idx_j]

                        P_jdx_j = np.where(masked_state_Ls_j == 3)[0]
                        if len(P_jdx_j) < min_trajectory_length:
                          continue
                        first_P_j = P_jdx_j[0]
                        last_P_j = P_jdx_j[-1]

                        masked_freqs_P_j = masked_freqs_j[first_P_j:last_P_j+1]
                        masked_times_P_j = masked_times_j[first_P_j:last_P_j+1]

                        delta_masked_freqs_P_j = masked_freqs_P_j[1:] - masked_freqs_P_j[:-1]
                        # delta_f = f_t_plus_1 - f_t
                        delta_masked_times_P_j = masked_times_P_j[:-1]

                        intersect_times = np.intersect1d(delta_masked_times_P_i, delta_masked_times_P_j)

                        if len(intersect_times)>=3:

                            intersect_idx_i = [np.where(delta_masked_times_P_i == intersect_time)[0][0] for intersect_time in intersect_times ]
                            intersect_delta_i = delta_masked_freqs_P_i[intersect_idx_i]

                            intersect_idx_j = [np.where(delta_masked_times_P_j == intersect_time)[0][0] for intersect_time in intersect_times ]
                            intersect_delta_j = delta_masked_freqs_P_j[intersect_idx_j]

                            if len(intersect_delta_i) != len(intersect_delta_j):
                                print(len(intersect_delta_j), len(intersect_delta_j))

                            r2 = stats.pearsonr(intersect_delta_i, intersect_delta_j)[0] ** 2
                            r2s_all.append(r2)

            r2s_all = np.asarray(r2s_all)
            ratio_f_all = np.asarray(ratio_f_all)
            abs_delta_f_all = np.asarray(abs_delta_f_all)
            #print(abs_delta_f_all[:10])
            abs_delta_f_all = abs_delta_f_all/pt.B_S_generation_dict[taxon][treatment]
            #print(abs_delta_f_all[:10])

            #r2s_obs_dict[treatment + taxon] = {}
            #r2s_obs_dict[treatment + taxon]['r2'] = r2s_all
            #r2s_obs_dict[treatment + taxon]['ratio_f'] = ratio_f_all
            #r2s_obs_dict[treatment + taxon]['abs_delta_f'] = abs_delta_f_all

            r2s_obs_dict[treatment][taxon] = {}
            r2s_obs_dict[treatment][taxon]['r2'] = r2s_all
            r2s_obs_dict[treatment][taxon]['ratio_f'] = ratio_f_all
            r2s_obs_dict[treatment][taxon]['abs_delta_f'] = abs_delta_f_all


    for taxon in pt.taxa:

        for treatment in pt.treatments:

            convergence_matrix = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % (treatment+taxon)))

            f_max_all = []
            #for population in populations:
            for replicate in pt.replicates:
                population = treatment + taxon + replicate
                for gene_name in sorted(convergence_matrix.keys()):

                    for t,L,f,f_max in convergence_matrix[gene_name]['mutations'][population]:

                        f_max_all.append(f_max)

            #fmax_dict[treatment][taxon] =  np.asarray(f_max_all)
            r2s_obs_dict[treatment][taxon]['f_max'] = np.asarray(f_max_all)

    with open(pt.get_path() + '/data/mutation_dynamics.pickle', 'wb') as handle:
        pickle.dump(r2s_obs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



#run_analyses()
with open(pt.get_path() + '/data/mutation_dynamics.pickle', 'rb') as handle:
    r2s_obs_dict = pickle.load(handle)


analyses = ['f_max', 'abs_delta_f', 'ratio_f']
# get KS distance
ks_dict = {}

for analysis in analyses:
    ks_dict[analysis] = {}
    p_value_list = []
    for treatment_idx, treatment in enumerate(pt.treatments):

        ks_dict[analysis][treatment] = {}
        D, p_value = stats.ks_2samp(r2s_obs_dict[treatment]['B'][analysis], r2s_obs_dict[treatment]['S'][analysis])
        ks_dict[analysis][treatment]['D'] = D
        ks_dict[analysis][treatment]['p_value'] = p_value

        p_value_list.append(p_value)



    reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(p_value_list, alpha=0.05, method='fdr_bh')
    for treatment_idx, treatment in enumerate(pt.treatments):
        ks_dict[analysis][treatment]['p_value_bh'] = pvals_corrected[treatment_idx]





fig = plt.figure(figsize = (4, 8))
gs = gridspec.GridSpec(nrows=3, ncols=1)

row_count = 0

ax_f_max = fig.add_subplot(gs[0, 0])
ax_delta_f = fig.add_subplot(gs[1, 0])
ax_ratio_f = fig.add_subplot(gs[2, 0])


#ax_f_max.text(-0.1, 1.07, pt.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_delta_f.transAxes)
#ax_delta_f.text(-0.1, 1.07, pt.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ratio_f.transAxes)
#ax_ratio_f.text(-0.1, 1.07, pt.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_ratio_f.transAxes)


axes = [ax_f_max, ax_delta_f, ax_ratio_f]

label_latex_dict = {'abs_delta_f': r'$\frac{\left | \Delta f \right |}{\Delta \tau}$',
                    'ratio_f': r'$\frac{f(t+ \delta t)}{f(t)}$',
                    'r2': r'$\rho^{2}_{M^{(i)}, M^{(j)} } $',
                    'f_max': r'$f_{max}$'}


label_text_dict = {'abs_delta_f': 'Absolute change in allele frequencies\nper-generation',
                    'ratio_f': 'Ratio of allele frequency changes',
                    'r2': 'Squared correlation between\nallele frequency trajectories',
                    'f_max': 'Maximum observed allele frequency'}



xlim_dict = {'abs_delta_f': [0.000001, 0.01],
            'ratio_f': [0.3, 300],
            'f_max': [ 0.09, 1.03 ]}


ylim_dict = {'abs_delta_f': [0.0005, 1.1],
            'ratio_f': [0.0005, 1.1],
            'f_max': [ 0.0008, 1.03 ]}

#xlims = [[0.000001, 0.01], [0.3, 300], [0.08, 1.02]]
#ylims = [[0.0005, 1.1], [0.0005, 1.1], [0.01, 1.1]]
#ylims_inset = [[-0.07, 0.87],[-0.05, 0.6],[-0.05, 0.6]]

ylim_inset_dict = {'abs_delta_f': [-0.07, 0.87],
            'ratio_f': [-0.05, 0.6],
            'f_max': [-0.05, 0.6]}


count = 0
for analysis, analysis_ax,  in zip(analyses, axes):

    ins_ks = inset_axes(analysis_ax, width="100%", height="100%", loc='lower right', bbox_to_anchor=(0.14,0.07,0.4,0.38), bbox_transform=analysis_ax.transAxes)

    for treatment_idx, treatment in enumerate(pt.treatments):
        for taxon in pt.taxa:

            f_max_array_sort = np.sort(r2s_obs_dict[treatment][taxon][analysis])
            cdf = 1-  np.arange(len(f_max_array_sort))/float(len(f_max_array_sort))

            analysis_ax.plot(f_max_array_sort, cdf, c =pt.get_colors(treatment), ls=pt.get_taxon_ls(taxon), lw=3, alpha=0.8)

        D_treatment = ks_dict[analysis][treatment]['D']

        marker_style = dict(color=pt.get_colors(treatment),
                            markerfacecoloralt='white',
                            markerfacecolor=pt.get_colors(treatment))

        ins_ks.plot(treatment_idx, D_treatment, markersize = 11, marker = 'o',  \
            linewidth=0.4,  alpha=1, fillstyle='left', zorder=2 , **marker_style)

        if ks_dict[analysis][treatment]['p_value_bh'] < 0.05:
            ins_ks.text(treatment_idx, D_treatment+0.06, '*', ha='center', fontsize=9)

    # finish the ins_ks fiddling
    ins_ks.tick_params(labelsize=5)
    ins_ks.tick_params(axis='both', which='major', pad=1)

    ins_ks.set_xlim([-0.5, 2.5])
    ins_ks.set_ylim(ylim_inset_dict[analysis])



    ins_ks.set_xticks([0, 1, 2])
    ins_ks.set_xticklabels(['1-day', '10-days', '100-days'],fontweight='bold' )
    ins_ks.set_ylabel("KS distance", fontsize=7)
    ins_ks.tick_params(axis='x', labelsize=5, length = 0)

    ins_ks.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    ins_ks.yaxis.set_tick_params(labelsize=4)


    analysis_ax.set_xlabel('%s, %s' % (label_text_dict[analysis], label_latex_dict[analysis] ) , fontsize = 11)
    analysis_ax.set_yscale('log', base=10)
    analysis_ax.set_xlim(xlim_dict[analysis])

    analysis_ax.set_ylim(ylim_dict[analysis])
    analysis_ax.xaxis.set_tick_params(labelsize=8)
    analysis_ax.yaxis.set_tick_params(labelsize=8)

    #if count != 2:
    analysis_ax.set_xscale('log', base=10)

    analysis_ax.set_ylabel('Fraction ' + r'$\geq$' + label_latex_dict[analysis], fontsize=11)


    if count == 0:
        legend_elements_taxon = [Line2D([0], [0], ls='--', color='k', lw=1.5, label= pt.latex_dict['B']),
                           Line2D([0], [0], ls=':', color='k', lw=1.5, label= pt.latex_dict['S'])]

        analysis_ax.legend(handles=legend_elements_taxon, loc='upper right', fontsize=6)

    if count == 1:
        analysis_ax.axvline(x=1, color='k', linestyle=':', alpha = 0.8, zorder=1)


    count+=1




ax_list = [ax_f_max, ax_delta_f, ax_ratio_f]

for ax_idx, ax in enumerate(ax_list):
    ax.text(-0.1, 1.07, pt.sub_plot_labels[ax_idx], fontsize=8, fontweight='bold', ha='center', va='center', transform=ax.transAxes)





fig_name = pt.get_path() + '/figs/trajectory_CDF_B_S.jpg'
fig.subplots_adjust(hspace=0.45)
fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
