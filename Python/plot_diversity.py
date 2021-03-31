from __future__ import division
import os, sys, json, math, itertools
import numpy as np

import  matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import statsmodels.stats.multitest as multitest

import phylo_tools as pt

#import scipy.stats as stats
#import statsmodels.api as sm
#import statsmodels.stats.api as sms
#import statsmodels.formula.api as smf
#from statsmodels.compat import lzip

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import scipy.stats as stats

#from sklearn.metrics import pairwise_distances
#from skbio.stats.ordination import pcoa

import parse_file
import timecourse_utils
import mutation_spectrum_utils


#from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles


#from sklearn.decomposition import PCA


np.random.seed(123456789)

treatments=pt.treatments
replicates = pt.replicates

t_max = 700
set_time = 500


legend_elements_fixed = [Line2D([0], [0], color = 'none', marker='o', label=pt.latex_dict['B'],
                        markerfacecolor='k', markersize=10),
                    Line2D([0], [0], marker='o', color='none', label=pt.latex_dict['S'],
                        markerfacecolor='w', markersize=10, markeredgewidth=2)]



sys.stderr.write("Loading mutation data...\n")

mutation_trajectories = {}
fixed_mutation_trajectories = {}
delta_mutation_trajectories = {}
prob_extinct = {}
#transit_times = {}
taxa = ['B', 'S']

for treatment in treatments:
    for taxon in taxa:
        for replicate in replicates:
            population = treatment + taxon + replicate
            sys.stderr.write("Processing %s...\n" % population)
            times, Ms, fixed_Ms = parse_file.get_mutation_fixation_trajectories(population)
            Ms = Ms[times<=t_max]
            if type(fixed_Ms) != np.float64:
                fixed_Ms = fixed_Ms[times<=t_max]

            times = times[times<=t_max]
            fixed_mutation_trajectories[population] = (times, fixed_Ms)
            mutation_trajectories[population] = (times,np.log10(Ms))
            delta_mutation_trajectories[population] = (times[1:], np.log10(Ms[1:]/Ms[:-1] ))

            prob_extinct[population] = parse_file.estimate_prob_extinction(population)

            #sys.stderr.write("analyzed %d mutations!\n" % len(Ms))

# fixed mutations t-test for B vs S at day 1 and 10 at day 600

for treatment in ['0', '1']:
    fixation_count_dict = {}
    for taxon in taxa:
        fixation_count_dict[taxon] = []
        for replicate in replicates:
            population = treatment + taxon + replicate
            fixation_idx = np.where(fixed_mutation_trajectories[population][0] == 600)
            if type(fixed_mutation_trajectories[population][1]) == np.float64:
                continue
            fixation_count_dict[taxon].append(fixed_mutation_trajectories[population][1][fixation_idx][0])


position_dict = { '0': {'B': {'V': (0.4, 0.25), 'K': (0.4, 0.11)}, 'S': {'V': (0.4, 0.18), 'K': (0.4, 0.04)}},
                    '1': {'B': {'V': (0.4, 0.25), 'K': (0.4, 0.11)}, 'S': {'V': (0.4, 0.18), 'K': (0.4, 0.04)}},
                    '2': {'B': {'V': (0.03, 0.95), 'K': (0.03, 0.81)}, 'S': {'V': (0.03, 0.88), 'K': (0.03, 0.74)}}}



def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    rounded = np.round(x * mags) / mags

    #if str(rounded)[-2:] == '.0':
    #    rounded = int(rounded)

    return rounded



def format_parameter(taxon, parameter, estiamte, std_error, sig_figs=3):
    # parameter = string
    if parameter == 'V_max':

        if taxon == 'B':
            #param = r'$\partial_{t}M |_{\mathrm{max, WT}} =$'
            param = r'$v_{\mathrm{max, WT}} =$'
        else:
            #param = r'$\partial_{t}M |_{\mathrm{max, \Delta}spo0A} =$'
            param = r'$v_{\mathrm{max, \Delta}$ \mathit{spo0A}} =$'

    else:

        if taxon == 'B':
            param = r'$K_{\mathrm{WT}} =$'
        else:
            param = r'$K_{\mathrm{\Delta}spo0A} =$'

    #estimate_sig_fig =  round(estiamte, sig_figs - int(math.floor(math.log10(abs(estiamte)))) - 1)
    #std_error_sig_fig =  round(std_error, sig_figs - int(math.floor(math.log10(abs(std_error)))) - 1)

    signif(std_error, sig_figs)

    return param + str(signif(estiamte, sig_figs)) +  r'$\pm$' + str(signif(std_error, sig_figs))



#fig = plt.figure(figsize = (15, 8))
fig = plt.figure(figsize = (17, 12 ))

row_count = 0

#gs = gridspec.GridSpec(nrows=6, ncols=5)
gs = gridspec.GridSpec(nrows=6, ncols=4)

ax_hyper_0 = fig.add_subplot(gs[0:2, 0])
ax_hyper_1 = fig.add_subplot(gs[2:4, 0])
ax_hyper_2 = fig.add_subplot(gs[4:6, 0])

ax_M_max = fig.add_subplot(gs[0:3, 1])
ax_K = fig.add_subplot(gs[3:6, 1])

ax_extinct = fig.add_subplot(gs[0:3, 2:4])
ax_fixed = fig.add_subplot(gs[3:6, 2:4])

#ax_regression_wt = fig.add_subplot(gs[0:3, 1:3])
#ax_regression_spo0a = fig.add_subplot(gs[3:6, 1:3])

#ax_fixed = fig.add_subplot(gs[0:3, 3:])
#ax_fmax = fig.add_subplot(gs[3:6, 3:])

#ax_fmax = fig.add_subplot(gs[3:6, 2:4])

#ins_ks = inset_axes(ax_fmax, width="100%", height="100%", loc='lower right', bbox_to_anchor=(0.12,0.07,0.4,0.38), bbox_transform=ax_fmax.transAxes)
#axes = [ax_hyper_0, ax_hyper_1, ax_hyper_2, ax_M_max, ax_K, ax_fixed, ax_fmax, ins_ks]
axes = [ax_hyper_0, ax_hyper_1, ax_hyper_2, ax_M_max, ax_K, ax_extinct, ax_fixed]

for ax_i_idx, ax_i in enumerate(axes):
    if ax_i_idx == len(axes)-1:
        fontsize = 10
    else:
        fontsize = 11

    if (ax_i_idx == 3) or (ax_i_idx == 4):
        x_text = -0.3
    else:
        x_text = -0.1

    ax_i.text(x_text, 1.07, pt.sub_plot_labels[ax_i_idx], fontsize=fontsize, fontweight='bold', ha='center', va='center', transform=ax_i.transAxes)



#ax_hyper_0.text(-0.1, 1.07, pt.sub_plot_labels[0], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_hyper_0.transAxes)
#ax_hyper_1.text(-0.1, 1.07, pt.sub_plot_labels[1], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_hyper_1.transAxes)
#ax_hyper_2.text(-0.1, 1.07, pt.sub_plot_labels[2], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_hyper_2.transAxes)
#ax_regression_wt.text(-0.1, 1.07, pt.sub_plot_labels[3], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_regression_wt.transAxes)
#ax_regression_spo0a.text(-0.1, 1.07, pt.sub_plot_labels[4], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_regression_spo0a.transAxes)
#ax_fixed.text(-0.1, 1.07, pt.sub_plot_labels[5], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_fixed.transAxes)
#ax_fmax.text(-0.1, 1.07, pt.sub_plot_labels[6], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_fmax.transAxes)
#ax_fixed.text(-0.1, 1.07, pt.sub_plot_labels[3], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_fixed.transAxes)
#ax_fmax.text(-0.1, 1.07, pt.sub_plot_labels[4], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_fmax.transAxes)



#ins_ks.text(-0.1, 1.07, pt.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ins_ks.transAxes)

legend_elements_hyper = [Line2D([0], [0], ls='--', color='k', lw=1.5, label= 'Fit: ' + r'$\mathrm{WT}$'),
                   Line2D([0], [0], ls=':', color='k', lw=1.5, label= 'Fit: ' + r'$\Delta \mathit{spo0A}$')]
ax_hyper_0.legend(handles=legend_elements_hyper, loc='upper right', fontsize=6.5)




ax_hyper_list = [ax_hyper_0, ax_hyper_1, ax_hyper_2]

# ax = fig.add_subplot(gs[taxon_idx*2:(taxon_idx*2)+2  , taxon_list_idx])

param_dict = {}

for treatment_idx, treatment in enumerate(treatments):

    #ax_t_vs_M = plt.subplot2grid((1, 3), (row_count, 0), colspan=1)
    ax_t_vs_M = ax_hyper_list[treatment_idx]

    for taxon_i, taxon in enumerate(taxa):

        treatment_taxon_populations = []

        Mts_all_list = []
        Ms_all_list = []

        Ms_0_list = []

        for replicate in replicates:

            population = treatment + taxon + replicate

            Mts,Ms = mutation_trajectories[population]

            Ms_0_list.append(Ms[np.where(Mts==100)[0][0]])


            ax_t_vs_M.plot(Mts, 10**Ms, 'o-',color= pt.get_colors(treatment), marker=pt.plot_species_marker(taxon), fillstyle=pt.plot_species_fillstyle(taxon), alpha=1, markersize=7,linewidth=3, markeredgewidth=1.5, zorder=1)

            Mts_all_list.append(Mts)
            Ms_all_list.append(Ms)

        Ms_0_mean = np.mean(Ms_0_list)

        Mts_all = np.concatenate(Mts_all_list)
        Ms_all = np.concatenate(Ms_all_list)

        Mts_shifted_all = Mts_all - min(Mts_all)

        Mts_shifted_range = np.linspace(0, max(Mts_shifted_all), num=1000)
        b0, K, V_max, z, ses_K, ses_V_max =  pt.fit_hyperbolic_michaelis_menten_best_parameters(Mts_shifted_all, Ms_all)
        predicted_Ms = pt.hyperbolic_michaelis_menten(Mts_shifted_range, b0, K, V_max)

        pt.fit_hyperbolic_michaelis_menten_scipy(Mts_all, Ms_all, Ms_0_mean)

        #if (treatment == '1') or (treatment=='2'):
        #    continue

        #b0, K, v_0, z, ses_K, ses_v_0 =  pt.fit_hyperbolic_fitness_notation_best_parameters(Mts_shifted_all, Ms_all)
        #predicted_Ms = pt.hyperbolic_fitness_notation(Mts_shifted_range, b0, K, v_0)

        ax_t_vs_M.plot(Mts_shifted_range+100, 10**predicted_Ms, ls=pt.linestyle_dict[taxon], color='k', alpha=1, markersize=7,linewidth=3, markeredgewidth=1.5, zorder=1)
        #ax_t_vs_M.plot(Mts_shifted_range+100, predicted_Ms, ls=pt.linestyle_dict[taxon], color='k', alpha=1, markersize=7,linewidth=3, markeredgewidth=1.5, zorder=1)

        generations_per_day = pt.get_B_S_generations(taxon, treatment, day_cutoff=1)

        #formatted_parameter_K = format_parameter(taxon, 'K', K, ses_K, sig_figs=3)
        #formatted_parameter_V_max = format_parameter(taxon, 'V_max', V_max, ses_V_max, sig_figs=3)


        #ax_t_vs_M.text(position_dict[treatment][taxon]['K'][0], position_dict[treatment][taxon]['K'][1], formatted_parameter_K, fontsize=6, transform=ax_t_vs_M.transAxes)
        #ax_t_vs_M.text(position_dict[treatment][taxon]['V'][0], position_dict[treatment][taxon]['V'][1], formatted_parameter_V_max, fontsize=6, transform=ax_t_vs_M.transAxes)

        K_generations = K * pt.get_per_day_generations(taxon, treatment)
        ses_K_generations = ses_K * pt.get_per_day_generations(taxon, treatment)

        print(treatment, taxon, 'V_max',  V_max, '+/-', ses_V_max)
        print(treatment, taxon, 'K',  K, '+/-', ses_K)
        print(treatment, taxon, 'K_gen',  K_generations, '+/-', ses_K_generations)

        if taxon not in param_dict:
            param_dict[taxon] = {}
        param_dict[taxon][treatment] = {}
        param_dict[taxon][treatment]['K_gen'] = K_generations
        param_dict[taxon][treatment]['ses_K_generations'] = ses_K_generations
        param_dict[taxon][treatment]['V_max'] = V_max
        param_dict[taxon][treatment]['ses_V_max'] = ses_V_max




    ax_t_vs_M.set_ylim([0.53 , 800])
    ax_t_vs_M.set_yscale('log', base=10)
    ax_t_vs_M.set_xlabel('Days, ' + r'$t$', fontsize = 13)
    ax_t_vs_M.set_ylabel('Mutations, ' + r'$M(t)$', fontsize = 13)
    ax_t_vs_M.xaxis.set_tick_params(labelsize=8)
    ax_t_vs_M.yaxis.set_tick_params(labelsize=8)


    if treatment=='0':
        ax_t_vs_M.text(0.15, 0.9, pt.treatment_label_dict[treatment], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_t_vs_M.transAxes)
    elif treatment == '1':
        ax_t_vs_M.text(0.2, 0.9, pt.treatment_label_dict[treatment], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_t_vs_M.transAxes)
    else:
        ax_t_vs_M.text(0.25  , 0.9, pt.treatment_label_dict[treatment], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_t_vs_M.transAxes)



    row_count += 1




# plot paramters
#ax_M_max = fig.add_subplot(gs[0:3, 1])
#ax_K = fig.add_subplot(gs[3:6, 1])



#V_max_list = []
#K_list = []
#y_axis_params = []
counts_params = 0
counts_list = []
for taxon in taxa:

    for treatment in treatments:

        v_max_ = param_dict[taxon][treatment]['V_max']
        k_ = param_dict[taxon][treatment]['K_gen']
        #y_axis_params.append(counts_params)

        #V_max_list.append(x_)
        #K_list.append(y_)

        ax_M_max.scatter(v_max_, counts_params, s = 200, \
            linewidth=3, facecolors=pt.get_scatter_facecolor(taxon, treatment), \
            edgecolors=pt.get_colors(treatment), marker=pt.plot_species_marker(taxon), \
            alpha=1, zorder=2)

        ax_K.scatter(k_, counts_params, s = 200, \
            linewidth=3, facecolors=pt.get_scatter_facecolor(taxon, treatment), \
            edgecolors=pt.get_colors(treatment), marker=pt.plot_species_marker(taxon), \
            alpha=1, zorder=2)

        counts_list.append(counts_params)
        counts_params += 1




#ax.set_xlim([0.8 , 10])

ax_K.set_xscale('log', base=10)


#y_ticks_params =  ['1\nday', '10\ndays', '100\ndays', '1\nday', '10\ndays', '100\ndays']
y_ticks_params =  ['1', '10', '100', '1', '10', '100']
#rotation=90,
ax_M_max.set_yticks(counts_list)
ax_M_max.set_yticklabels( y_ticks_params, fontsize=8, va="center")

ax_K.set_yticks(counts_list)
ax_K.set_yticklabels( y_ticks_params, fontsize=8, va="center")


ax_M_max.set_xlabel('Maximum value of\n' + r'$M(t)$' + ', ' + r'$\left [ \mathrm{log}_{10} M \right ]_{\mathrm{max}}$', fontsize = 12)
ax_K.set_xlabel('Generation where ' + r'$M(t)$' + ' is\nhalf its maximum, '  + r'$\tau_{1/2}$', fontsize = 12)




ax_M_max.tick_params(axis="x", labelsize=7.5)
ax_K.tick_params(axis="x", labelsize=8)


ax_M_max.set_ylim([-0.5, max(counts_list)+0.5])
ax_K.set_ylim([-0.5, max(counts_list)+0.5])

#ax_M_max.tick_params(labelsize=8)

ax_M_max.set_xlim([0.3, 8.8])

ax_K.set_xlim([10**1.1, 10**3.7])


ax_M_max.text(-0.23, 0.5, "Day(s)", fontsize=14, rotation=90, ha='center', va='center', transform=ax_M_max.transAxes)
ax_K.text(-0.23, 0.5, "Day(s)", fontsize=14, rotation=90, ha='center', va='center', transform=ax_K.transAxes)




ax_M_max.axhline(y=2.5, color='k', linestyle='--',  lw=2, alpha=1, zorder=1)
ax_K.axhline(y=2.5, color='k', linestyle='--',  lw=2, alpha=1, zorder=1)


ax_M_max.legend(handles=legend_elements_fixed, loc='upper right', fontsize=8)
ax_K.legend(handles=legend_elements_fixed, loc='upper right', fontsize=8)


# plot probability of extinction

count = 0
for treatment in pt.treatments:
    for taxon in pt.taxa:
        fixed_mutations_i = [y for x,y in prob_extinct.items() if treatment+taxon in x]

        ax_extinct.scatter(list(itertools.repeat(count, len(fixed_mutations_i))), fixed_mutations_i, s = 180, \
            linewidth=3, facecolors=pt.get_scatter_facecolor(taxon, treatment), \
            edgecolors=pt.get_colors(treatment), marker=pt.plot_species_marker(taxon), \
            alpha=0.6, zorder=2)

        count+=1


ax_extinct.set_ylabel(r'$\mathrm{P\left [ Extinct | \, detected, \right ]}$', fontsize=14)
#ax_fixed.text(-0.1, 1.01, pt.sub_plot_labels[sub_plot_counts], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_fixed.transAxes)

#ax_pca.set_xscale('symlog')
ax_extinct.set_xlim([-0.5, 5.5])

#ax_fixed.axhline(y=0, color='k', linestyle='--',  lw=2, alpha=1, zorder=1)


ax_extinct.set_xticks([])
# for minor ticks
ax_extinct.set_xticks([], minor=True)
ax_extinct.set_xticklabels([])
ax_extinct.set_xticks([0.5, 2.5, 4.5])
ax_extinct.set_xticklabels(['1-day', '10-days', '100-days'])
ax_extinct.tick_params(axis='x', labelsize=14, length = 0)
ax_extinct.tick_params(axis="y", labelsize=6)


ax_extinct.legend(handles=legend_elements_fixed, loc='upper left', fontsize=8)









# then plot fixed mutations
fixed_mutations_per_generation = {}
for taxon in ['B', 'S']:
    for treatment in treatments:
        for replicate in pt.replicates:
            population = treatment + taxon + replicate
            #if population in pt.populations_to_ignore:
            #    continue

            times, Ms, fixed_Ms = parse_file.get_mutation_fixation_trajectories(population)

            if isinstance(fixed_Ms,float) == True:
                fixed_Ms = np.asarray([0]* len(times))

            #fixed_mutations_per_day[population] = (fixed_Ms[-1]+(int(times[-1])/1000))/int(times[-1])
            generations_per_day = pt.get_B_S_generations(taxon, treatment, day_cutoff=1)
            fixed_mutations_per_generation[population] = fixed_Ms[-1]/( times[-1] * generations_per_day)



count = 0
for treatment in pt.treatments:
    for taxon in pt.taxa:
        fixed_mutations_i = [y for x,y in fixed_mutations_per_generation.items() if treatment+taxon in x]

        ax_fixed.scatter(list(itertools.repeat(count, len(fixed_mutations_i))), fixed_mutations_i, s = 180, \
            linewidth=3, facecolors=pt.get_scatter_facecolor(taxon, treatment), \
            edgecolors=pt.get_colors(treatment), marker=pt.plot_species_marker(taxon), \
            alpha=0.6, zorder=2)

        count+=1

ax_fixed.set_ylabel('Fixed mutations, per-generation', fontsize=14)
#ax_fixed.text(-0.1, 1.01, pt.sub_plot_labels[sub_plot_counts], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_fixed.transAxes)

#ax_pca.set_xscale('symlog')
ax_fixed.set_xlim([-0.5, 5.5])

ax_fixed.axhline(y=0, color='k', linestyle='--',  lw=2, alpha=1, zorder=1)


ax_fixed.set_xticks([])
# for minor ticks
ax_fixed.set_xticks([], minor=True)
ax_fixed.set_xticklabels([])


ax_fixed.set_xticks([0.5, 2.5, 4.5])
ax_fixed.set_xticklabels(['1-day', '10-days', '100-days'])
ax_fixed.tick_params(axis='x', labelsize=14, length = 0)

#ax_fixed.tick_params(labelsize=8)

ax_fixed.tick_params(axis="y", labelsize=6)




ax_fixed.legend(handles=legend_elements_fixed, loc='upper left', fontsize=8)



# get fmax
#fmax_dict = {}
# loop through taxa and get M(700) for all reps in each treatment
#for treatment in pt.treatments:
#    fmax_dict[treatment] = {}




#ks_dict = {}
#treatment_ = []
#p_values = []
#for treatment in treatments:

#    ks_dict[treatment] = {}

#    sample_1 = fmax_dict[treatment]['B']
#    sample_2 = fmax_dict[treatment]['S']

#    D, p_value = stats.ks_2samp(sample_1, sample_2)

#    ks_dict[treatment]['D'] = D
#    ks_dict[treatment]['p_value'] = p_value

#    #treatment_pairs.append((treatment_pair, taxon))
#    p_values.append(p_value)


#reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(p_values, alpha=0.05, method='fdr_bh')
#for treatment_idx, treatment in enumerate(treatments):
#    ks_dict[treatment]['p_value_bh'] = pvals_corrected[treatment_idx]


#for treatment in pt.treatments:

#    for taxon, f_max_array in fmax_dict[treatment].items():

#        f_max_array_sort = np.sort(f_max_array)
#        cdf = 1-  np.arange(len(f_max_array_sort))/float(len(f_max_array_sort))
#        #num_bins = 40
#        #counts, bin_edges = np.histogram(f_max_array_sort, bins=num_bins, normed=True)
#        #cdf = np.cumsum(counts)
#        #pylab.plot(bin_edges[1:], cdf)
#        ax_fmax.plot(f_max_array_sort, cdf, c =pt.get_colors(treatment), ls=pt.get_taxon_ls(taxon), lw=3, alpha=0.8)
#        #marker=pt.plot_species_marker(taxon), markersize=1)



#ax_fmax.set_xlim([ 0.09, 1.03 ])
#ax_fmax.set_ylim([ 0.0008, 1.03 ])

#ax_fmax.set_xscale('log', base=10)
#ax_fmax.set_yscale('log', base=10)
#ax_fmax.tick_params(labelsize=8)

#ax_fmax.set_xlabel('Maximum observed allele frequency, ' + r'$f_{max}$', fontsize=12)
#ax_fmax.set_ylabel('Fraction of mutations ' + r'$\geq f_{max}$', fontsize=12)


#legend_elements_fmax = [Line2D([0], [0], ls='--', color='k', lw=1.5, label= pt.latex_dict['B']),
#                   Line2D([0], [0], ls=':', color='k', lw=1.5, label= pt.latex_dict['S'])]

#ax_fmax.legend(handles=legend_elements_fmax, loc='upper right', fontsize=8)

#ax_fmax.tick_params(axis="y", labelsize=6)



#ins_ks.set_xlabel('Max.' + r'$\left \langle x(t) \right \rangle$', fontsize=8)
#ins_ks.set_ylabel("KS distance", fontsize=8)



#mean_D_dict = {}
#for treatment_idx, treatment in enumerate(ks_dict.keys()):

#    D = ks_dict[treatment]['D']

#    marker_style = dict(color=pt.get_colors(treatment),
#                        markerfacecoloralt='white',
#                        markerfacecolor=pt.get_colors(treatment))

#    ins_ks.plot(treatment_idx, D, markersize = 11, marker = 'o',  \
#        linewidth=0.4,  alpha=1, fillstyle='left', zorder=2 , **marker_style)

#    if ks_dict[treatment]['p_value_bh'] < 0.05:
#        ins_ks.text(treatment_idx, D+0.06, '*', ha='center', fontsize=8)



#ins_ks.tick_params(labelsize=5)
#ins_ks.tick_params(axis='both', which='major', pad=1)

#ins_ks.set_xlim([-0.5, 2.5])
#ins_ks.set_ylim([-0.05, 0.6])

#ins_ks.set_xticks([0, 1, 2])
#ins_ks.set_xticklabels(['1-day', '10-days', '100-days'],fontweight='bold')
#ins_ks.tick_params(axis='x', labelsize=6.5, length = 0)

#ins_ks.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)


#fig.text(0.53, 0.02, 'Days, ' + r'$t$', ha='center', fontsize=28)

fig.subplots_adjust(hspace=0.8, wspace=0.4) #hspace=0.3, wspace=0.5
fig_name = pt.get_path() + '/figs/diversity.jpg'
# pad_inches = 0.4,
fig.savefig(fig_name, format='jpg',  bbox_inches = "tight", pad_inches = 0.1, dpi = 600)
plt.close()
