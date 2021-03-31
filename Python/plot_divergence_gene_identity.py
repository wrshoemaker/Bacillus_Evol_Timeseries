from __future__ import division
import os, sys, pickle, random
import numpy as np
from itertools import combinations

import  matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.colors import ColorConverter

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.stats as stats
import statsmodels.api as sm

import parse_file
import timecourse_utils
import mutation_spectrum_utils
import phylo_tools as pt


np.random.seed(123456789)
random.seed(123456789)

permutations_gene_content_divergence = 10000
#permutations_gene_content_divergence = 10

permutations_divergence = 10000

# permutations for anova
n_permutations = 100000
#n_permutations = 10
treatment_pairs = [['0','1'],['0','2'],['1','2']]


gene_data = parse_file.parse_gene_list('B')
gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data


#standardized_gene_overlap_bw_treatments = {}
#standardized_gene_overlap_bw_taxa = {}
enriched_gene_dict = {}
for taxon in pt.taxa:

    enriched_gene_dict[taxon] = {}

    #gene_dict = {}
    #N_significant_genes_dict = {}

    for treatment in pt.treatments:

        genes_significant_file_path = pt.get_path() +'/data/timecourse_final/' +  ("parallel_%ss_%s.txt" % ('gene', treatment+taxon))
        genes_nonsignificant_file_path = pt.get_path() +'/data/timecourse_final/' +  ("parallel_not_significant_%ss_%s.txt" % ('gene', treatment+taxon))

        if os.path.exists(genes_significant_file_path) == False:
            continue

        genes_significant_file = open(genes_significant_file_path, 'r')
        first_line_significant = genes_significant_file.readline()

        N_significant_genes = 0

        genes = []

        for line in genes_significant_file:
            line_split = line.strip().split(', ')
            gene_name = line_split[0]
            genes.append(gene_name)
            N_significant_genes += 1

        genes_significant_file.close()

        enriched_gene_dict[taxon][treatment] = {}
        enriched_gene_dict[taxon][treatment] = set(genes)

        #N_significant_genes_dict[treatment] = N_significant_genes
        #gene_dict[treatment] = set(genes)




standardized_gene_overlap_bw_treatments = {}
standardized_gene_overlap_bw_taxa = {}

for taxon in pt.taxa:

    standardized_gene_overlap_bw_treatments[taxon] = {}

    for treatment_pair in combinations(pt.treatments, 2):

        N_genes_1 = len(enriched_gene_dict[taxon][treatment_pair[0]])
        N_genes_2 = len(enriched_gene_dict[taxon][treatment_pair[1]])
        jaccard_treatment_pair = len(enriched_gene_dict[taxon][treatment_pair[0]] & enriched_gene_dict[taxon][treatment_pair[1]]) / len(enriched_gene_dict[taxon][treatment_pair[0]] | enriched_gene_dict[taxon][treatment_pair[1]])

        jaccard_null = []

        for i in range(permutations_gene_content_divergence):

            sample_1 = set(random.sample(gene_names, N_genes_1))
            sample_2 = set(random.sample(gene_names, N_genes_2))

            jaccard_treatment_pair_i = len(sample_1 & sample_2) / len(sample_1 | sample_2)

            jaccard_null.append(jaccard_treatment_pair_i)

        jaccard_null = np.asarray(jaccard_null)

        standardized_jaccard = (jaccard_treatment_pair-np.mean(jaccard_null)) / np.std(jaccard_null)

        P_jaccard = (len(jaccard_null[jaccard_null>jaccard_treatment_pair]) + 1) / (permutations_gene_content_divergence+1)

        standardized_gene_overlap_bw_treatments[taxon][treatment_pair] = {}
        standardized_gene_overlap_bw_treatments[taxon][treatment_pair]['jaccard'] = jaccard_treatment_pair
        standardized_gene_overlap_bw_treatments[taxon][treatment_pair]['Z_jaccard'] = standardized_jaccard
        standardized_gene_overlap_bw_treatments[taxon][treatment_pair]['P'] = P_jaccard



for treatment in pt.treatments:

    N_genes_1 = len(enriched_gene_dict['B'][treatment])
    N_genes_2 = len(enriched_gene_dict['S'][treatment])
    jaccard_treatment_pair = len(enriched_gene_dict['B'][treatment] & enriched_gene_dict['S'][treatment]) / len(enriched_gene_dict['B'][treatment] | enriched_gene_dict['S'][treatment])

    jaccard_null = []
    for i in range(permutations_gene_content_divergence):

        sample_1 = set(random.sample(gene_names, N_genes_1))
        sample_2 = set(random.sample(gene_names, N_genes_2))

        jaccard_treatment_pair_i = len(sample_1 & sample_2) / len(sample_1 | sample_2)

        jaccard_null.append(jaccard_treatment_pair_i)

    jaccard_null = np.asarray(jaccard_null)

    standardized_jaccard = (jaccard_treatment_pair-np.mean(jaccard_null)) / np.std(jaccard_null)

    P_jaccard = (len(jaccard_null[jaccard_null>jaccard_treatment_pair]) + 1) / (permutations_gene_content_divergence+1)

    standardized_gene_overlap_bw_taxa[treatment] = {}
    standardized_gene_overlap_bw_taxa[treatment]['jaccard'] = jaccard_treatment_pair
    standardized_gene_overlap_bw_taxa[treatment]['Z_jaccard'] = standardized_jaccard
    standardized_gene_overlap_bw_taxa[treatment]['P'] = P_jaccard







gs = gridspec.GridSpec(nrows=2, ncols=1)

fig = plt.figure(figsize = (10, 13))
ax_between_taxa = fig.add_subplot(gs[0, 0])
ax_between_treatments = fig.add_subplot(gs[1, 0])

#ax_between_taxa.set_title("Convergent/divergent evolution as the\nproportion of shared enriched genes ", fontsize=16, fontweight='bold')


ax_between_taxa.text(-0.1, 1.07, pt.sub_plot_labels[0], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_between_taxa.transAxes)
ax_between_treatments.text(-0.1, 1.07, pt.sub_plot_labels[1], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)



for treatment_idx, treatment in enumerate(standardized_gene_overlap_bw_taxa.keys()):

    Z_jaccard = standardized_gene_overlap_bw_taxa[treatment]['Z_jaccard']

    marker_style = dict(color=pt.get_colors(treatment),
                        markerfacecoloralt='white',
                        markerfacecolor=pt.get_colors(treatment),
                        mew=3)

    ax_between_taxa.plot(treatment, Z_jaccard, markersize = 30, marker = 'o',  \
        linewidth=0.4,  alpha=1, fillstyle='left', zorder=2 , **marker_style)

    if standardized_gene_overlap_bw_taxa[treatment]['P'] < 0.05:
        ax_between_taxa.text(treatment, Z_jaccard+2, '*', ha='center', fontweight='bold', fontsize=20)






ax_between_taxa.set_xlim([-0.5,2.5])
ax_between_taxa.set_ylim([-3 ,50])


ax_between_taxa.axhline( y=0, color='k', lw=3, linestyle=':', alpha = 1, zorder=1)

ax_between_taxa.arrow(0.06, 0.65, 0.0, 0.2, width=0.012,fc='k', ec='k', transform=ax_between_taxa.transAxes)
ax_between_taxa.text(0.06, 0.5, 'Increasing\nconvergence', fontsize=12, fontweight='bold', ha='center', va='center', rotation=90, transform=ax_between_taxa.transAxes)



#ax_between_taxa.set_ylabel("Standardized correlation, "+ r'$Z_{\rho}$' , fontsize = 16)
ax_between_taxa.set_ylabel("Standardized fraction of\nshared enriched genes", fontsize = 16)



ax_between_taxa.set_xticks([0, 1, 2])
ax_between_taxa.set_xticklabels(['1-day', '10-days', '100-days'], fontweight='bold', fontsize=18 )



ax_between_taxa.set_title("B. subtilis " + r'$\mathbf{WT}$' + " vs. B. subtilis "  + r'$\mathbf{\Delta spo0A}$', style='italic', fontsize=16, fontweight='bold')




count = 0
for taxon in pt.taxa:
    for treatment_pair_idx, treatment_pair in enumerate(treatment_pairs):

        Z_jaccard = standardized_gene_overlap_bw_treatments[taxon][tuple(treatment_pair)]['Z_jaccard']


        marker_style = dict(color='k', marker='o',
                markerfacecoloralt=pt.get_colors(treatment_pair[1]),
                markerfacecolor=pt.get_colors(treatment_pair[0]),
                mew=2)

        ax_between_treatments.plot(count, Z_jaccard, markersize = 28,   \
            linewidth=2,  alpha=1, zorder=3, fillstyle='left', **marker_style)

        if standardized_gene_overlap_bw_treatments[taxon][tuple(treatment_pair)]['P'] < 0.05:
            ax_between_treatments.text(count, Z_jaccard+2.2, '*', ha='center', fontweight='bold', fontsize=20)

        count+=1



ax_between_treatments.set_xticks([0, 1, 2, 3, 4, 5])

ax_between_treatments.set_xticklabels(['1-day vs.\n10-days', '1-day vs.\n100-days', '10-days vs.\n100-days', '1-day vs.\n10-days', '1-day vs.\n100-days', '10-days vs.\n100-days'], fontweight='bold', fontsize=13 )


ax_between_treatments.axhline( y=0, color='k', lw=3, linestyle=':', alpha = 1, zorder=1)
ax_between_treatments.axvline( x=2.5, color='k', lw=3, linestyle='-', alpha = 1, zorder=1)

ax_between_treatments.arrow(0.06, 0.80, 0.0, 0.1, width=0.012,fc='k', ec='k', transform=ax_between_treatments.transAxes)
ax_between_treatments.text(0.06, 0.65, 'Increasing\nconvergence', fontsize=12, fontweight='bold', ha='center', va='center', rotation=90, transform=ax_between_treatments.transAxes)

ax_between_treatments.set_xlim([-0.5,5.5])
ax_between_treatments.set_ylim([-3,75])

#ax_between_treatments.set_ylabel("Standardized correlation, "+ r'$Z_{\rho}$' , fontsize = 16)
ax_between_treatments.set_ylabel("Standardized fraction of\nshared enriched genes", fontsize = 16)


ax_between_treatments.text(0.25, -0.155, "B. subtilis " + r'$\mathbf{WT}$', style='italic', fontsize=16, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)
ax_between_treatments.text(0.75, -0.155, "B. subtilis "  + r'$\mathbf{\Delta spo0A}$', style='italic', fontsize=16, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)


fig.subplots_adjust(hspace=0.15,wspace=0.2) #hspace=0.3, wspace=0.5
fig_name = pt.get_path() + "/figs/divergence_gene_identity.jpg"
fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



sys.stderr.write("Done with figure!\n")
