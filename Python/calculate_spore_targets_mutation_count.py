from __future__ import division
import numpy
import random
import os
import sys
import parse_file
import mutation_spectrum_utils
import timecourse_utils
import phylo_tools as pt

import phik

n_samples = 10000

numpy.random.seed(123456789)

treatments=pt.treatments
replicates = pt.replicates



spore_locus_tags = parse_file.get_spore_locus_tags()

gene_data = parse_file.parse_gene_list('B')

gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data

n_samples = 10000


count_dict = {}
L_all = 0.0
L_spore = 0.0
count = 0
for treatment_idx, treatment in enumerate(pt.treatments):
    count_dict[treatment] = {}
    for taxon in pt.taxa:
        convergence_matrix = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % (treatment+taxon)))
        populations = [treatment+taxon + replicate for replicate in replicates ]
        gene_parallelism_statistics = mutation_spectrum_utils.calculate_parallelism_statistics(convergence_matrix,populations,Lmin=100)

        N_muts = sum([gene_parallelism_statistics[locus_tag]['observed'] for locus_tag in gene_parallelism_statistics.keys()])
        N_muts_spore = sum([gene_parallelism_statistics[locus_tag]['observed'] for locus_tag in gene_parallelism_statistics.keys() if locus_tag in spore_locus_tags])

        if count == 0:
            for locus_tag in gene_parallelism_statistics.keys():
                L_all += gene_parallelism_statistics[locus_tag]['length']
                if locus_tag in spore_locus_tags:
                    L_spore += gene_parallelism_statistics[locus_tag]['length']


        count_dict[treatment][taxon] = {}
        count_dict[treatment][taxon]['N_muts'] = N_muts
        count_dict[treatment][taxon]['N_muts_spore'] = N_muts_spore

        count += 1



# try with contingency table

for treatment_idx, treatment in enumerate(pt.treatments):

    N_muts_wt = count_dict[treatment]['B']['N_muts']
    N_muts_wt_spore = count_dict[treatment]['B']['N_muts_spore']

    N_muts_spo0a = count_dict[treatment]['S']['N_muts']
    N_muts_spo0a_spore = count_dict[treatment]['S']['N_muts_spore']

    #table = numpy.asarray([[N_muts_wt_spore, N_muts_wt - N_muts_wt_spore], [N_muts_spo0a_spore, N_muts_spo0a - N_muts_spo0a_spore]])


    #odds_ratio = (table[0,0] * table[1,1]) / (table[1,0] * table[0,1])
    #odds_ratio_null = []
    #for i in range(n_samples):
    #    table_null = phik.simulation.sim_2d_data_patefield(table)
    #    odds_ratio_null.append((table_null[0,0] * table_null[1,1]) / (table_null[1,0] * table_null[0,1]))

    #odds_ratio_null = numpy.asarray(odds_ratio_null)
    #P = (sum(odds_ratio_null>odds_ratio)+1) / (n_samples+1)

    #delta_ell

    diff = (N_muts_wt_spore/N_muts_wt) - (N_muts_spo0a_spore/N_muts_spo0a)
    abs_diff = numpy.absolute(diff)

    null_abs_diff = []

    for i in range(n_samples):

        N_muts_wt_spore_null, N_muts_wt_not_spore_null = numpy.random.multinomial(N_muts_wt, [L_spore/L_all, 1 - (L_spore/L_all)])
        N_muts_spo0a_spore_null, N_muts_spo0a_not_spore_null = numpy.random.multinomial(N_muts_spo0a, [L_spore/L_all, 1 - (L_spore/L_all)])

        fract_spore_wt_null = N_muts_wt_spore_null / (N_muts_wt_spore_null + N_muts_wt_not_spore_null)
        fract_spore_spo0a_null = N_muts_spo0a_spore_null / (N_muts_spo0a_spore_null + N_muts_spo0a_not_spore_null)
        null_abs_diff.append( numpy.absolute(fract_spore_wt_null - fract_spore_spo0a_null) )

    null_abs_diff = numpy.asarray(null_abs_diff)

    P = (sum(null_abs_diff>abs_diff)+1) / (len(null_abs_diff)+1)

    if diff>0:
        winner = 'WT'
    else:
        winner = 'spo0A'

    print(winner, abs_diff, P)





#cG_subsample_list = []
