from __future__ import division
import os, sys, pickle, random
import numpy as np

import  matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.colors import ColorConverter

import scipy.stats as stats
import statsmodels.stats.multitest as multitest

import parse_file
import timecourse_utils
import mutation_spectrum_utils
import phylo_tools as pt

#import get_random_matrix

import phik

np.random.seed(123456789)


treatments = pt.treatments

# to-do: re-do analysis for enriched genes in *either* treatment you're comparing
# read in nonsignificant genes and add those conts in..


permutations_divergence = 100000

treatment_pairs = [['0','1'],['0','2'],['1','2']]


gene_data_B = parse_file.parse_gene_list('B')
gene_names_B, gene_start_positions_B, gene_end_positions_B, promoter_start_positions_B, promoter_end_positions_B, gene_sequences_B, strands_B, genes_B, features_B, protein_ids_B = gene_data_B
gene_name_dict = dict(zip(gene_names_B, genes_B ))
protein_id_dict = dict(zip(gene_names_B, protein_ids_B ))

significant_multiplicity_dict = {}
significant_n_mut_dict = {}
gene_size_dict = {}
gene_mean_size_dict = {}


for taxon in pt.taxa:
    significant_multiplicity_dict[taxon] = {}
    significant_n_mut_dict[taxon] = {}
    gene_size_dict[taxon] = {}

    gene_data = parse_file.parse_gene_list(taxon)

    gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data


    convergence_matrix = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % ('0'+taxon)))
    Ltot = 0
    for gene_name in sorted(convergence_matrix.keys()):
        Lmin=0
        L = max([convergence_matrix[gene_name]['length'],Lmin])
        Ltot += L
    Lavg = Ltot*1.0/len(convergence_matrix.keys())

    gene_mean_size_dict[taxon] = Lavg

    for treatment_idx, treatment in enumerate(pt.treatments):

        significant_multiplicity_taxon_path = pt.get_path() + '/data/timecourse_final/parallel_genes_%s.txt' % (treatment+taxon)
        if os.path.exists(significant_multiplicity_taxon_path) == False:
            continue
        significant_multiplicity_taxon = open(significant_multiplicity_taxon_path, "r")
        for i, line in enumerate( significant_multiplicity_taxon ):
            if i == 0:
                continue
            line = line.strip()
            items = line.split(",")
            gene_size_dict[taxon][items[0]] = float(items[-5])
            if items[0] not in significant_multiplicity_dict[taxon]:
                significant_multiplicity_dict[taxon][items[0]] = {}

            if items[0] not in significant_n_mut_dict[taxon]:
                significant_n_mut_dict[taxon][items[0]] = {}

            significant_multiplicity_dict[taxon][items[0]][treatment] = float(items[-2])
            significant_n_mut_dict[taxon][items[0]][treatment] = float(items[-4])





def calculate_divergence_correlations_between_taxa():

    sys.stdout.write("Starting divergence tests...\n")

    #output_file = open(pt.get_path() + "/data/divergent_genes_between_taxa.txt", "w")
    # print header
    #output_file.write(", ".join(["Transfer regime", "Taxon", "Locus tag", "RefSeq protein ID", "Gene", "|Delta relative mult|", "P BH corrected"]))

    divergence_dict = {}
    all_p_value_corr = []
    all_p_value_corr_squared = []
    all_p_value_mean_abs_diff = []
    for treatment_idx, treatment in enumerate(pt.treatments):
        all_genes = set(significant_n_mut_dict['B'].keys()) & significant_n_mut_dict['S'].keys()
        result = []
        for gene in all_genes:
            if (treatment in significant_n_mut_dict['B'][gene]) and (treatment in significant_n_mut_dict['S'][gene]):
                result.append((significant_n_mut_dict['B'][gene][treatment], significant_n_mut_dict['S'][gene][treatment], gene ))

        n_x = [int(x[0]) for x in result]
        n_y = [int(x[1]) for x in result]
        gene_names = [x[2] for x in result]

        gene_sizes_taxon = [gene_size_dict['B'][gene_i] for gene_i in gene_names]
        gene_sizes_taxon = np.asarray(gene_sizes_taxon)
        taxon_Lmean = gene_mean_size_dict['B']

        n_matrix = np.asarray([n_x, n_y])
        mult_matrix = n_matrix * (taxon_Lmean / gene_sizes_taxon)
        rel_mult_matrix = mult_matrix/mult_matrix.sum(axis=1)[:,None]
        pearsons_corr = np.corrcoef(rel_mult_matrix[0,:], rel_mult_matrix[1,:])[1,0]

        mean_abs_diff = np.mean( np.absolute(rel_mult_matrix[0,:] - rel_mult_matrix[1,:]))

        pearsons_corr_squared = pearsons_corr**2
        pearsons_corr_null = []
        pearsons_corr_squared_null = []

        mean_abs_diff_null = []

        obs_rel_difference = rel_mult_matrix[0,:] - rel_mult_matrix[1,:]

        gene_obs_rel_difference_null_dict = {}

        for k in range(permutations_divergence):

            if (k % 2000 == 0) and (k>0):

                sys.stdout.write("%d iterations\n" % (k))

            n_matrix_random = phik.simulation.sim_2d_data_patefield(n_matrix)
            mult_matrix_random = n_matrix_random * (taxon_Lmean / gene_sizes_taxon)
            rel_mult_matrix_random = mult_matrix_random/mult_matrix_random.sum(axis=1)[:,None]
            pearsons_corr_random = np.corrcoef(rel_mult_matrix_random[0,:], rel_mult_matrix_random[1,:])[1,0]

            mean_absolute_difference = np.mean( np.absolute(rel_mult_matrix_random[0,:] - rel_mult_matrix_random[1,:]))

            pearsons_corr_squared_random = pearsons_corr_random**2

            pearsons_corr_null.append(pearsons_corr_random)
            pearsons_corr_squared_null.append(pearsons_corr_squared_random)
            mean_abs_diff_null.append(mean_absolute_difference)

            #obs_rel_difference_null = rel_mult_matrix_random[0,:] - rel_mult_matrix_random[1,:]
            #for gene_name_i, obs_rel_difference_null_i in zip(gene_names, obs_rel_difference_null):
            #    if gene_name_i not in gene_obs_rel_difference_null_dict:
            #        gene_obs_rel_difference_null_dict[gene_name_i] = []
            #    gene_obs_rel_difference_null_dict[gene_name_i].append(obs_rel_difference_null_i)

        #p_value_list = []
        #for gene_name_i, obs_rel_difference_i in zip(gene_names, obs_rel_difference):
        #    obs_rel_difference_abs_i = np.absolute(obs_rel_difference_i)
        #    obs_rel_difference_null_array_i = np.asarray(gene_obs_rel_difference_null_dict[gene_name_i])
        #    obs_rel_difference_null_array_abs_i = np.absolute(obs_rel_difference_null_array_i)
        #    p_value_i = (len(obs_rel_difference_null_array_abs_i[obs_rel_difference_null_array_abs_i>obs_rel_difference_abs_i]) +1) / (permutations_divergence+1)
        #    p_value_list.append(p_value_i)

        #reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(p_value_list, alpha=0.05, method='fdr_bh')
        #for gene_name_i_idx, gene_name_i in enumerate(gene_names):

        #    obs_rel_difference_i = obs_rel_difference[gene_name_i_idx]
        #    p_value_bh_i = pvals_corrected[gene_name_i_idx]

        #    if p_value_bh_i>=0.05:
        #        continue

        #    if obs_rel_difference_i > 0:
        #        taxon_i = 'B'
        #    else:
        #        taxon_i = 'S'

        #    obs_rel_difference_abs_i = np.absolute(obs_rel_difference_i)

        #    output_file.write("\n")
        #    output_file.write("%d, %s, %s, %s, %s, %f, %f" % (int(10**int(treatment)), taxon_i, gene_name_i, protein_id_dict[gene_name_i], gene_name_dict[gene_name_i], obs_rel_difference_abs_i, p_value_bh_i))


        pearsons_corr_null = np.asarray(pearsons_corr_null)
        pearsons_corr_squared_null = np.asarray(pearsons_corr_squared_null)

        pearsons_corr_null_abs = np.absolute(pearsons_corr_null)
        pearsons_corr_squared_null_abs = np.absolute(pearsons_corr_squared_null)

        mean_abs_diff_null = np.asarray(mean_abs_diff_null)

        Z_corr_squared = pt.calculate_standard_score(pearsons_corr_squared, pearsons_corr_squared_null)
        Z_corr = pt.calculate_standard_score(pearsons_corr, pearsons_corr_null)

        Z_mean_abs_diff = pt.calculate_standard_score(mean_abs_diff, mean_abs_diff_null)

        #P_corr_squared = (len(pearsons_corr_squared_null_abs[pearsons_corr_squared_null_abs < np.absolute(pearsons_corr_squared)]) +1) / (permutations_divergence+1)
        #P_corr = (len(pearsons_corr_null_abs[pearsons_corr_null_abs < np.absolute(pearsons_corr)]) +1) / (permutations_divergence+1)

        #_corr_squared = (len(pearsons_corr_squared_null_abs[pearsons_corr_squared_null_abs < np.absolute(pearsons_corr_squared)]) +1) / (permutations_divergence+1)
        #P_corr = (len(pearsons_corr_null_abs[pearsons_corr_null_abs < np.absolute(pearsons_corr)]) +1) / (permutations_divergence+1)

        P_corr_squared = pt.calculate_p_value_permutation(pearsons_corr_squared, pearsons_corr_squared_null)
        P_corr = pt.calculate_p_value_permutation(pearsons_corr, pearsons_corr_null)

        P_mean_abs_diff = pt.calculate_p_value_permutation(mean_abs_diff, mean_abs_diff_null)

        divergence_dict[treatment] = {}
        divergence_dict[treatment]['pearsons_corr_squared'] = pearsons_corr_squared
        divergence_dict[treatment]['P_value_corr_squared'] = P_corr_squared
        divergence_dict[treatment]['Z_corr_squared'] = Z_corr_squared

        divergence_dict[treatment]['pearsons_corr'] = pearsons_corr
        divergence_dict[treatment]['P_value_corr'] = P_corr
        divergence_dict[treatment]['Z_corr'] = Z_corr

        divergence_dict[treatment]['mean_abs_diff'] = mean_abs_diff
        divergence_dict[treatment]['P_mean_abs_diff'] = P_mean_abs_diff
        divergence_dict[treatment]['Z_mean_abs_diff'] = Z_mean_abs_diff

        all_p_value_corr.append(P_corr)
        all_p_value_corr_squared.append(P_corr_squared)

        all_p_value_mean_abs_diff.append(P_mean_abs_diff)

        sys.stdout.write("%d-day, WT vs. spo0A: MAD=%f, P=%f, Z=%f\n" % (10**int(treatment), mean_abs_diff, P_mean_abs_diff, Z_mean_abs_diff))

    reject_corr, pvals_corrected_corr, alphacSidak_corr, alphacBonf_corr = multitest.multipletests(all_p_value_corr, alpha=0.05, method='fdr_bh')
    reject_corr_squared, pvals_corrected_corr_squared, alphacSidak_corr_squared, alphacBonf_corr_squared = multitest.multipletests(all_p_value_corr_squared, alpha=0.05, method='fdr_bh')
    reject_mad, pvals_corrected_mad, alphacSidak_mad, alphacBonf_mad = multitest.multipletests(all_p_value_mean_abs_diff, alpha=0.05, method='fdr_bh')

    for treatment_idx, treatment in enumerate(pt.treatments):

        divergence_dict[treatment]['P_value_corr_bh'] = pvals_corrected_corr[treatment_idx]
        divergence_dict[treatment]['P_value_corr_squared_bh'] = pvals_corrected_corr_squared[treatment_idx]
        divergence_dict[treatment]['P_value_mean_abs_diff_bh'] = pvals_corrected_mad[treatment_idx]

    sys.stdout.write("Dumping pickle......\n")

    sys.stdout.write("Saving divergent genes......\n")
    #output_file.close()
    with open(pt.get_path()+'/data/divergence_pearsons_between_taxa.pickle', 'wb') as handle:
        pickle.dump(divergence_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.write("Done!\n")






def calculate_divergence_correlations_between_treatments():

    sys.stdout.write("Starting divergence tests...\n")

    divergence_dict = {}

    all_p_value_corr = []
    all_p_value_corr_squared = []
    all_p_value_mean_abs_diff = []

    for treatment_pair_idx, treatment_pair in enumerate(treatment_pairs):

        treatment_pair_set = (treatment_pair[0], treatment_pair[1])

        divergence_dict[treatment_pair_set] = {}

        for taxon in pt.taxa:

            #result = [(x[treatment_pair[0]],x[treatment_pair[1]]) for x in significant_multiplicity_dict[taxon].values() if (treatment_pair[0] in x) and (treatment_pair[1] in x)]
            #result = [(x[treatment_pair[0]],x[treatment_pair[1]], x) for x in significant_n_mut_dict[taxon].values() if (treatment_pair[0] in x) and (treatment_pair[1] in x)]
            result = [(dicts[treatment_pair[0]],dicts[treatment_pair[1]], keys) for keys, dicts in significant_n_mut_dict[taxon].items() if (treatment_pair[0] in dicts) and (treatment_pair[1] in dicts)]

            n_x = [int(x[0]) for x in result]
            n_y = [int(x[1]) for x in result]
            gene_names = [x[2] for x in result]

            gene_sizes_taxon_treatment_pair = [gene_size_dict[taxon][gene_i] for gene_i in gene_names]
            gene_sizes_taxon_treatment_pair = np.asarray(gene_sizes_taxon_treatment_pair)
            taxon_Lmean = gene_mean_size_dict[taxon]

            n_matrix = np.asarray([n_x, n_y])
            mult_matrix = n_matrix * (taxon_Lmean / gene_sizes_taxon_treatment_pair)
            rel_mult_matrix = mult_matrix/mult_matrix.sum(axis=1)[:,None]
            pearsons_corr = np.corrcoef(rel_mult_matrix[0,:], rel_mult_matrix[1,:])[1,0]
            pearsons_corr_squared = pearsons_corr**2

            mean_abs_diff = np.mean( np.absolute(rel_mult_matrix[0,:] - rel_mult_matrix[1,:]))

            pearsons_corr_null = []
            pearsons_corr_squared_null = []
            mean_abs_diff_null = []
            for k in range(permutations_divergence):

                if (k % 2000 == 0) and (k>0):

                    sys.stdout.write("%d iterations\n" % (k))

                n_matrix_random = phik.simulation.sim_2d_data_patefield(n_matrix)
                mult_matrix_random = n_matrix_random * (taxon_Lmean / gene_sizes_taxon_treatment_pair)
                rel_mult_matrix_random = mult_matrix_random/mult_matrix_random.sum(axis=1)[:,None]
                pearsons_corr_random = np.corrcoef(rel_mult_matrix_random[0,:], rel_mult_matrix_random[1,:])[1,0]
                pearsons_corr_squared_random = pearsons_corr_random**2

                mean_absolute_difference = np.mean( np.absolute(rel_mult_matrix_random[0,:] - rel_mult_matrix_random[1,:]))

                pearsons_corr_null.append(pearsons_corr_random)
                pearsons_corr_squared_null.append(pearsons_corr_squared_random)
                mean_abs_diff_null.append(mean_absolute_difference)

            pearsons_corr_null = np.asarray(pearsons_corr_null)
            pearsons_corr_squared_null = np.asarray(pearsons_corr_squared_null)

            pearsons_corr_null_abs = np.absolute(pearsons_corr_null)
            pearsons_corr_squared_null_abs = np.absolute(pearsons_corr_squared_null)

            mean_abs_diff_null = np.asarray(mean_abs_diff_null)


            Z_corr_squared = pt.calculate_standard_score(pearsons_corr_squared, pearsons_corr_squared_null)
            Z_corr = pt.calculate_standard_score(pearsons_corr, pearsons_corr_null)

            Z_mean_abs_diff = pt.calculate_standard_score(mean_abs_diff, mean_abs_diff_null)

            #P_corr_squared = (len(pearsons_corr_squared_null_abs[pearsons_corr_squared_null_abs < np.absolute(pearsons_corr_squared)]) +1) / (permutations_divergence+1)
            #P_corr = (len(pearsons_corr_null_abs[pearsons_corr_null_abs < np.absolute(pearsons_corr)]) +1) / (permutations_divergence+1)

            P_corr_squared = pt.calculate_p_value_permutation(pearsons_corr_squared, pearsons_corr_squared_null)
            P_corr = pt.calculate_p_value_permutation(pearsons_corr, pearsons_corr_null)

            P_mean_abs_diff = pt.calculate_p_value_permutation(mean_abs_diff, mean_abs_diff_null)

            divergence_dict[treatment_pair_set][taxon] = {}
            divergence_dict[treatment_pair_set][taxon]['pearsons_corr_squared'] = pearsons_corr_squared
            divergence_dict[treatment_pair_set][taxon]['P_value_corr_squared'] = P_corr_squared
            divergence_dict[treatment_pair_set][taxon]['Z_corr_squared'] = Z_corr_squared

            divergence_dict[treatment_pair_set][taxon]['pearsons_corr'] = pearsons_corr
            divergence_dict[treatment_pair_set][taxon]['P_value_corr'] = P_corr
            divergence_dict[treatment_pair_set][taxon]['Z_corr'] = Z_corr

            divergence_dict[treatment_pair_set][taxon]['mean_abs_diff'] = mean_abs_diff
            divergence_dict[treatment_pair_set][taxon]['P_mean_abs_diff'] = P_mean_abs_diff
            divergence_dict[treatment_pair_set][taxon]['Z_mean_abs_diff'] = Z_mean_abs_diff


            all_p_value_corr.append(P_corr)
            all_p_value_corr_squared.append(P_corr_squared)
            all_p_value_mean_abs_diff.append(P_mean_abs_diff)


            sys.stdout.write("%d vs %d-day, %s: MAD=%f, P=%f, Z=%f\n" % (10**int(treatment_pair[0]), 10**int(treatment_pair[1]), taxon, mean_abs_diff, P_mean_abs_diff, Z_mean_abs_diff))


    reject_corr, pvals_corrected_corr, alphacSidak_corr, alphacBonf_corr = multitest.multipletests(all_p_value_corr, alpha=0.05, method='fdr_bh')
    reject_corr_squared, pvals_corrected_corr_squared, alphacSidak_corr_squared, alphacBonf_corr_squared = multitest.multipletests(all_p_value_corr_squared, alpha=0.05, method='fdr_bh')
    reject_mad, pvals_corrected_mad, alphacSidak_mad, alphacBonf_mad = multitest.multipletests(all_p_value_mean_abs_diff, alpha=0.05, method='fdr_bh')

    count_multitest = 0

    for treatment_pair in treatment_pairs:

        treatment_pair_set = (treatment_pair[0], treatment_pair[1])

        for taxon in pt.taxa:

            divergence_dict[treatment_pair_set][taxon]['P_value_corr_bh'] = pvals_corrected_corr[count_multitest]
            divergence_dict[treatment_pair_set][taxon]['P_value_corr_squared_bh'] = pvals_corrected_corr_squared[count_multitest]
            divergence_dict[treatment_pair_set][taxon]['P_value_mean_abs_diff_bh'] = pvals_corrected_mad[count_multitest]

            count_multitest+=1

    sys.stdout.write("Dumping pickle......\n")
    with open(pt.get_path()+'/data/divergence_pearsons_between_treatments.pickle', 'wb') as handle:
        pickle.dump(divergence_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.write("Done!\n")


#calculate_divergence_correlations_between_taxa()
#calculate_divergence_correlations_between_treatments()

with open(pt.get_path()+'/data/divergence_pearsons_between_taxa.pickle', 'rb') as handle:
    divergence_dict_between_taxa = pickle.load(handle)

with open(pt.get_path()+'/data/divergence_pearsons_between_treatments.pickle', 'rb') as handle:
    divergence_dict_between_treatments = pickle.load(handle)


fitness_dict = pt.get_fitness_dict()




gs = gridspec.GridSpec(nrows=2, ncols=2)

fig = plt.figure(figsize = (10, 13))
#ax_between_taxa = fig.add_subplot(gs[0, 0])
#ax_between_treatments = fig.add_subplot(gs[1, 0])

ax_between_treatments = fig.add_subplot(gs[0, 0:])
ax_between_taxa = fig.add_subplot(gs[1, 0])
ax_between_taxa_vs_fitness = fig.add_subplot(gs[1, 1])


ax_between_treatments.text(-0.1, 1.07, pt.sub_plot_labels[0], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)
ax_between_taxa.text(-0.1, 1.07, pt.sub_plot_labels[1], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_between_taxa.transAxes)
ax_between_taxa_vs_fitness.text(-0.1, 1.07, pt.sub_plot_labels[2], fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_between_taxa_vs_fitness.transAxes)


offset_dict = {'0': {'x_start':0.65 , 'y_start':0.3 , 'x_end': -0.6, 'y_end':-0.12},
                '1': {'x_start':-0.25 , 'y_start':-0.5  , 'x_end': 0.4, 'y_end':0.4}}


for treatment_idx, treatment in enumerate(divergence_dict_between_taxa.keys()):

    Z_corr = divergence_dict_between_taxa[treatment]['Z_mean_abs_diff']

    marker_style = dict(color=pt.get_colors(treatment),
                        markerfacecoloralt='white',
                        markerfacecolor=pt.get_colors(treatment),
                        mew=3)

    ax_between_taxa.plot(treatment, Z_corr, markersize = 30, marker = 'o',  \
        linewidth=0.4,  alpha=1, fillstyle='left', zorder=2 , **marker_style)

    if divergence_dict_between_taxa[treatment]['P_value_mean_abs_diff_bh'] < 0.05:
        ax_between_taxa.text(treatment, Z_corr+0.7, '*', ha='center', fontweight='bold', fontsize=18)

    mean_fitness = fitness_dict['mean_fitness'][treatment]
    se_fitness = fitness_dict['se_fitness'][treatment]

    ax_between_taxa_vs_fitness.plot(Z_corr, mean_fitness, markersize = 20, marker = 'o',  \
        linewidth=0.4,  alpha=1, fillstyle='left', zorder=2 , **marker_style)


    #ax_between_taxa_vs_fitness.errorbar(Z_corr, mean_fitness, se_fitness, linestyle='-', marker='o', c=pt.get_colors(treatment), lw = 3, zorder=2)
    ax_between_taxa_vs_fitness.errorbar(Z_corr, mean_fitness, se_fitness, linestyle='-', fmt='none', ecolor=pt.get_colors(treatment), lw = 3, zorder=2)


    if treatment_idx == 2:
        continue

    # plot arrow
    Z_corr_dt = divergence_dict_between_taxa[treatments[treatment_idx+1]]['Z_mean_abs_diff']
    mean_fitness_dt = fitness_dict['mean_fitness'][treatments[treatment_idx+1]]

    delta_Z = Z_corr_dt - Z_corr
    delta_mean_fitness = mean_fitness_dt - mean_fitness


    # looks like shit, fix it later
    prop = dict(arrowstyle="-|>, head_width=0.6,head_length=1.2",
            shrinkA=0,shrinkB=0, facecolor='k', lw=3)

    ax_between_taxa_vs_fitness.annotate("", xy=(Z_corr_dt+offset_dict[treatment]['x_end'], mean_fitness_dt+offset_dict[treatment]['y_end']), xytext=(Z_corr+offset_dict[treatment]['x_start'], mean_fitness+offset_dict[treatment]['y_start']), arrowprops=prop)



    #ax_between_taxa_vs_fitness.arrow(Z_corr, mean_fitness, delta_Z, delta_mean_fitness, head_width=0.5, head_length=0.7, lw=2, length_includes_head=True, fc='k', ec='k')









ax_between_taxa.set_xlim([-0.5,2.5])
ax_between_taxa.set_ylim([-2,12])


ax_between_taxa.axhline( y=0, color='k', lw=3, linestyle=':', alpha = 1, zorder=1)
#ax_between_taxa.text(0.125, 0.91, 'Convergence', fontsize=15, fontweight='bold', ha='center', va='center', transform=ax_between_taxa.transAxes)
#ax_between_taxa.text(0.115, 0.83, 'Divergence', fontsize=15 , fontweight='bold', ha='center', va='center', transform=ax_between_taxa.transAxes)

ax_between_taxa.text(0.5, 0.19, 'Divergence', fontsize=15, fontweight='bold', ha='center', va='center', transform=ax_between_taxa.transAxes)
ax_between_taxa.text(0.5, 0.10, 'Convergence', fontsize=15 , fontweight='bold', ha='center', va='center', transform=ax_between_taxa.transAxes)


ax_between_taxa.set_ylabel("Standardized mean absolute difference\nin mutation counts among genes, "+ r'$Z_{\left \langle \Delta \mathcal{M} \right \rangle}$' , fontsize = 16)

ax_between_taxa.set_xticks([0, 1, 2])
ax_between_taxa.set_xticklabels(['1-day', '10-days', '100-days'], fontweight='bold', fontsize=14 )

ax_between_taxa.set_title("B. subtilis " + r'$\mathbf{WT}$' + " vs. "  + r'$\mathbf{\Delta spo0A}$', style='italic', fontsize=16, fontweight='bold')



ax_between_taxa_vs_fitness.set_xlabel( r'$\mathrm{WT}$' + " vs. "  + r'$\mathrm{\Delta spo0A}$' + ' ' +  r'$Z_{\left \langle \Delta \mathcal{M} \right \rangle}$', fontsize = 16)
ax_between_taxa_vs_fitness.set_ylabel("Fitness of " + r'$\Delta \mathit{spo0A}$' + ' after ' + r'$t$' + ' days, ' + r'$X(t)$'  , fontsize = 16)
ax_between_taxa_vs_fitness.set_xlim([1,12])
ax_between_taxa_vs_fitness.axhline( y=0, color='k', lw=3, linestyle=':', alpha = 1, label='Neutrality', zorder=1)
ax_between_taxa_vs_fitness.legend(loc='upper left', fontsize=11)



count = 0
for taxon in pt.taxa:
    for treatment_pair_idx, treatment_pair in enumerate(treatment_pairs):

        Z_corr = divergence_dict_between_treatments[tuple(treatment_pair)][taxon]['Z_mean_abs_diff']

        marker_style = dict(color='k', marker='o',
                markerfacecoloralt=pt.get_colors(treatment_pair[1]),
                markerfacecolor=pt.get_colors(treatment_pair[0]),
                mew=2)

        ax_between_treatments.plot(count, Z_corr, markersize = 28,   \
            linewidth=2,  alpha=1, zorder=3, fillstyle='left', **marker_style)

        if divergence_dict_between_treatments[tuple(treatment_pair)][taxon]['P_value_mean_abs_diff_bh'] < 0.05:
            ax_between_treatments.text(count, Z_corr+0.7, '*', ha='center', fontweight='bold', fontsize=20)

        count+=1



ax_between_treatments.set_xticks([0, 1, 2, 3, 4, 5])

ax_between_treatments.set_xticklabels(['1-day vs.\n10-days', '1-day vs.\n100-days', '10-days vs.\n100-days', '1-day vs.\n10-days', '1-day vs.\n100-days', '10-days vs.\n100-days'], fontweight='bold', fontsize=13 )

ax_between_treatments.axhline( y=0, color='k', lw=3, linestyle=':', alpha = 1, zorder=1)
ax_between_treatments.axvline( x=2.5, color='k', lw=3, linestyle='-', alpha = 1, zorder=1)

ax_between_treatments.text(0.115, 0.21, 'Divergence', fontsize=15, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)
ax_between_treatments.text(0.125, 0.12, 'Convergence', fontsize=15 , fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)
ax_between_treatments.set_xlim([-0.5,5.5])
ax_between_treatments.set_ylim([-2.8,14])

#ax_between_treatments.set_ylabel("Standardized correlation, "+ r'$Z_{\rho}$' , fontsize = 16)
ax_between_treatments.set_ylabel("Standardized mean absolute difference\nin mutation counts among genes, "+ r'$Z_{\left \langle \Delta \mathcal{M} \right \rangle}$' , fontsize = 16)

#\left \langle \Delta \mathcal{M} \right \rangle

ax_between_treatments.text(0.25, -0.155, "B. subtilis " + r'$\mathbf{WT}$', style='italic', fontsize=16, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)
ax_between_treatments.text(0.75, -0.155, "B. subtilis "  + r'$\mathbf{\Delta spo0A}$', style='italic', fontsize=16, fontweight='bold', ha='center', va='center', transform=ax_between_treatments.transAxes)


fig.subplots_adjust(hspace=0.3,wspace=0.2) #hspace=0.3, wspace=0.5
fig_name = pt.get_path() + "/figs/divergence_mutation_counts.pdf"
fig.savefig(fig_name, format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



sys.stderr.write("Done with figure!\n")
