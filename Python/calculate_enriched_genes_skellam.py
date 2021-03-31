from __future__ import division

import sys, pickle, math, random, os, itertools, re
from itertools import combinations
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

import numpy as np

import scipy.stats as stats

import statsmodels.stats.multitest as mt

import parse_file
import timecourse_utils
import mutation_spectrum_utils
import phylo_tools as pt

import mpmath

from scipy.special import iv

n_min=3

treatment_pairs = [['0','1'],['0','2'],['1','2']]


def skellam_pdf(delta_n, lambda_1, lambda_2):

    if delta_n > 0:

        pmf = ((lambda_1/lambda_2)**(delta_n/2)) * iv(delta_n, 2*np.sqrt(lambda_1*lambda_2))
        pmf += ((lambda_2/lambda_1)**(delta_n/2)) * iv(-1*delta_n, 2*np.sqrt(lambda_1*lambda_2))
        pmf *= np.exp((-1*lambda_1) + (-1*lambda_2))

        #pmf = ((lambda_1/lambda_2)**(delta_n/2)) * mpmath.besseli(delta_n, 2*np.sqrt(lambda_1*lambda_2), zeroprec=80, infprec=80)
        #pmf += ((lambda_2/lambda_1)**(delta_n/2)) * mpmath.besseli(-1*delta_n, 2*np.sqrt(lambda_1*lambda_2), zeroprec=80, infprec=80)
        #pmf *= np.exp((-1*lambda_1) + (-1*lambda_2))

    else:

        pmf = np.exp((-1*lambda_1) + (-1*lambda_2)) * iv(0, 2*np.sqrt(lambda_1*lambda_2))
        #pmf = np.exp((-1*lambda_1) + (-1*lambda_2)) * mpmath.besseli(0, 2*np.sqrt(lambda_1*lambda_2), zeroprec=80, infprec=80)


    return pmf



def calculate_survival(counts_1, counts_2, genes, n_min = 3, alpha = 0.05):

    #counts_1 = np.asarray([8,3,4,0,0,0,2,2,7,2,4,2,0,2,2,0,2,2,2,0,4,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,0,2,2,2,0,2,2])
    #counts_2 = np.asarray([0,8,2,0,1,0,3,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,3,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,0,2,2,2,0,2,2])

    lambda_1 = sum(counts_1)/len(counts_1)
    lambda_2 = sum(counts_2)/len(counts_2)

    delta_n_original = np.absolute(counts_1-counts_2)
    delta_n = delta_n_original[delta_n_original>n_min]

    delta_n_no_absolute = counts_1-counts_2
    delta_n_no_absolute = delta_n_no_absolute[delta_n_original>n_min]

    genes = genes[delta_n_original>n_min]

    delta_n_range = list(range(0,500))
    delta_n_range_array = np.asarray(delta_n_range)



    delta_n_range_array_subset = delta_n_range_array[delta_n_range_array<=max(delta_n)]

    pmf = [skellam_pdf(i, lambda_1, lambda_2) for i in delta_n_range]
    pmf = np.asarray(pmf)

    survival_null = [ 1-sum(pmf[:i]) for i in range(len(pmf)) ]
    survival_null = np.asarray(survival_null)
    survival_null = survival_null[delta_n_range_array<=max(delta_n)]

    survival_obs = [ len(delta_n[delta_n>=i])/len(delta_n) for i in delta_n_range]
    survival_obs = np.asarray(survival_obs)
    survival_obs = survival_obs[delta_n_range_array<=max(delta_n)]

    P_values = [sum(pmf[delta_n_range.index(delta_n_i):]) for delta_n_i in delta_n]
    P_values = np.asarray(P_values)

    expected_number_genes = 0
    P_range = np.linspace(10**-4, 0.05, num=10000)[::-1]

    N_bar_P_star_div_N_P_star = []
    P_stars = []
    N_bar_P = []
    for P_range_i in P_range:

        N_P_star = len(P_values[P_values<P_range_i])
        N_bar_P_star = 0

        if N_P_star == 0:
            continue

        #for g in range(len(counts_1)):
        for delta_n_j_idx, delta_n_j in enumerate(delta_n):
            if delta_n_j < n_min:
                continue

            P_delta_n_j = sum(pmf[delta_n_range.index(delta_n_j):])

            if P_range_i > P_delta_n_j:
                # no gene specific indices so just multiply the final probability by number of genes
                N_bar_P_star += skellam_pdf(delta_n_j, lambda_1, lambda_2) * len(delta_n_original)

        #print(P_range_i, N_bar_P_star)
        N_bar_P_star_div_N_P_star.append(N_bar_P_star/N_P_star)
        P_stars.append(P_range_i)

        #N_bar_P.append(N_bar_P_star)

    N_bar_P_star_div_N_P_star = np.asarray(N_bar_P_star_div_N_P_star)
    P_stars = np.asarray(P_stars)

    position_P_star = np.argmax(N_bar_P_star_div_N_P_star<=0.05)

    P_star = P_stars[position_P_star]

    #N_bar_P = np.asarray(N_bar_P)
    return delta_n_range_array_subset, genes, survival_obs, survival_null, delta_n_no_absolute, P_values, P_star





gene_data = parse_file.parse_gene_list('B')

gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data


def calculate_between_taxa():

    between_taxa_dict = {}
    for treatment_idx, treatment in enumerate(pt.treatments):

        convergence_matrix_B = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % (treatment + 'B')))
        populations_B = [treatment+'B'+replicate for replicate in pt.replicates ]
        gene_parallelism_statistics_B = mutation_spectrum_utils.calculate_parallelism_statistics(convergence_matrix_B,populations_B,Lmin=100)

        convergence_matrix_S = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % (treatment + 'S')))
        populations_S = [treatment+'S'+replicate for replicate in pt.replicates ]
        gene_parallelism_statistics_S = mutation_spectrum_utils.calculate_parallelism_statistics(convergence_matrix_S,populations_S,Lmin=100)

        n_muts_B = []
        n_muts_S = []

        locus_tags = np.asarray(list(gene_parallelism_statistics_B.keys()))

        for locus_tag in locus_tags:
            n_muts_B.append(gene_parallelism_statistics_B[locus_tag]['observed'])
            n_muts_S.append(gene_parallelism_statistics_S[locus_tag]['observed'])

        n_muts_B = np.asarray(n_muts_B)
        n_muts_S = np.asarray(n_muts_S)

        delta_n_range_array_subset, genes_keep, survival_obs, survival_null, delta_n, P_values, P_star = calculate_survival(n_muts_B, n_muts_S, locus_tags, n_min=n_min)

        # now print to file
        between_taxa_dict[treatment] = {}
        between_taxa_dict[treatment]['delta_n'] = delta_n
        between_taxa_dict[treatment]['P_values'] = P_values
        between_taxa_dict[treatment]['genes_keep'] = genes_keep


    sys.stdout.write("Saving divergent genes......\n")
    with open(pt.get_path()+'/data/divergence_skellam_between_taxa.pickle', 'wb') as handle:
        pickle.dump(between_taxa_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.write("Done!\n")






def calculate_between_treatments():

    between_treatment_dict = {}

    for taxon in pt.taxa:

        between_treatment_dict[taxon] = {}

        for treatment_pair_idx, treatment_pair in enumerate(treatment_pairs):

            convergence_matrix_1 = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % (treatment_pair[0] + taxon)))
            populations_1 = [treatment_pair[0]+taxon+replicate for replicate in pt.replicates ]
            gene_parallelism_statistics_1 = mutation_spectrum_utils.calculate_parallelism_statistics(convergence_matrix_1,populations_1,Lmin=100)

            convergence_matrix_2 = parse_file.parse_convergence_matrix(pt.get_path() + '/data/timecourse_final/' +("%s_convergence_matrix.txt" % (treatment_pair[1] + taxon)))
            populations_2 = [treatment_pair[1]+taxon+replicate for replicate in pt.replicates ]
            gene_parallelism_statistics_2 = mutation_spectrum_utils.calculate_parallelism_statistics(convergence_matrix_2,populations_2,Lmin=100)

            n_muts_1 = []
            n_muts_2 = []

            locus_tags = np.asarray(list(gene_parallelism_statistics_1.keys()))

            for locus_tag in locus_tags:
                n_muts_1.append(gene_parallelism_statistics_1[locus_tag]['observed'])
                n_muts_2.append(gene_parallelism_statistics_2[locus_tag]['observed'])

            n_muts_1 = np.asarray(n_muts_1)
            n_muts_2 = np.asarray(n_muts_2)

            delta_n_range_array_subset, genes_keep, survival_obs, survival_null, delta_n, P_values, P_star = calculate_survival(n_muts_1, n_muts_2, locus_tags, n_min=n_min)

            # now print to file
            treatment_pair_set = (treatment_pair[0], treatment_pair[1])
            between_treatment_dict[taxon][treatment_pair_set] = {}
            between_treatment_dict[taxon][treatment_pair_set]['delta_n'] = delta_n
            between_treatment_dict[taxon][treatment_pair_set]['P_values'] = P_values
            between_treatment_dict[taxon][treatment_pair_set]['genes_keep'] = genes_keep


    sys.stdout.write("Saving divergent genes......\n")
    with open(pt.get_path()+'/data/divergence_skellam_between_treatments.pickle', 'wb') as handle:
        pickle.dump(between_treatment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.write("Done!\n")



def get_unique_diverged_genes_between_taxa_old():

    with open(pt.get_path()+'/data/divergence_skellam_between_taxa.pickle', 'rb') as handle:
        divergence_dict_between_taxa = pickle.load(handle)

    treatments_set = set(pt.treatments)

    output_file_bw_taxa = open(pt.get_path() + "/data/divergence_skellam_between_taxa.txt", "w")
    output_file_bw_taxa.write(", ".join(["Transfer regime", "Strain", "Locus tag", "delta_n", "-log10[P]"]))

    for treatment in divergence_dict_between_taxa.keys():
        other_treatments = treatments_set - set(treatment)

        delta_n = divergence_dict_between_taxa[treatment]['delta_n']
        genes_keep = divergence_dict_between_taxa[treatment]['genes_keep']
        P_values = divergence_dict_between_taxa[treatment]['P_values']

        delta_n_wt = delta_n[delta_n>0]
        genes_keep_wt = genes_keep[delta_n>0]
        P_values_wt = P_values[delta_n>0]

        delta_n_spo0a = delta_n[delta_n<0]
        genes_keep_spo0a = genes_keep[delta_n<0]
        P_values_spo0a = P_values[delta_n<0]

        genes_keep_set_wt = set(genes_keep_wt)
        genes_keep_set_spo0a = set(genes_keep_spo0a)

        treatment_label = '%d-day' % 10**int(treatment)

        for other_treatment in other_treatments:

            delta_n_other_treatment = divergence_dict_between_taxa[other_treatment]['delta_n']
            genes_keep_other_treatment = divergence_dict_between_taxa[other_treatment]['genes_keep']
            P_values_other_treatment = divergence_dict_between_taxa[other_treatment]['P_values']

            delta_n_other_treatment_wt = delta_n_other_treatment[delta_n_other_treatment>0]
            genes_keep_other_treatment_wt = genes_keep_other_treatment[delta_n_other_treatment>0]
            P_values_other_treatment_wt = P_values_other_treatment[delta_n_other_treatment>0]

            delta_n_other_treatment_spo0a = delta_n_other_treatment[delta_n_other_treatment<0]
            genes_keep_other_treatment_spo0a = genes_keep_other_treatment[delta_n_other_treatment<0]
            P_values_other_treatment_spo0a = P_values_other_treatment[delta_n_other_treatment<0]

            genes_keep_set_wt -= set(genes_keep_other_treatment_wt)
            genes_keep_set_wt -= set(genes_keep_other_treatment_spo0a)

            genes_keep_set_spo0a -= set(genes_keep_other_treatment_spo0a)
            genes_keep_set_spo0a -= set(genes_keep_other_treatment_wt)


        genes_keep_unique_wt = np.asarray(list(genes_keep_set_wt))
        genes_keep_unique_idx_wt = [np.where(genes_keep_wt == k)[0][0] for k in genes_keep_unique_wt]
        genes_keep_unique_idx_wt = np.asarray(genes_keep_unique_idx_wt)
        if len(genes_keep_unique_idx_wt) > 0 :

            genes_keep_unique_wt = genes_keep_wt[genes_keep_unique_idx_wt]
            delta_n_unique_wt = delta_n_wt[genes_keep_unique_idx_wt]
            P_values_unique_wt = P_values_wt[genes_keep_unique_idx_wt]

            delta_n_unique_wt = np.absolute(delta_n_unique_wt)
            P_values_unique_wt = np.log10(P_values_unique_wt)*-1

            for g in range(len(genes_keep_unique_wt)):

                output_file_bw_taxa.write("\n")
                output_file_bw_taxa.write(", ".join([treatment_label, 'WT', genes_keep_unique_wt[g], str(delta_n_unique_wt[g]), str(round(P_values_unique_wt[g], 5))]))


        genes_keep_unique_spo0a = np.asarray(list(genes_keep_set_spo0a))
        genes_keep_unique_idx_spo0a = [np.where(genes_keep_spo0a == k)[0][0] for k in genes_keep_unique_spo0a]
        genes_keep_unique_idx_spo0a = np.asarray(genes_keep_unique_idx_spo0a)
        if len(genes_keep_unique_idx_spo0a) > 0 :

            genes_keep_unique_spo0a = genes_keep_spo0a[genes_keep_unique_idx_spo0a]
            delta_n_unique_spo0a = delta_n_spo0a[genes_keep_unique_idx_spo0a]
            P_values_unique_spo0a = P_values_spo0a[genes_keep_unique_idx_spo0a]

            delta_n_unique_spo0a = np.absolute(delta_n_unique_spo0a)
            P_values_unique_spo0a = np.log10(P_values_unique_spo0a)*-1

            for g in range(len(genes_keep_unique_spo0a)):

                output_file_bw_taxa.write("\n")
                output_file_bw_taxa.write(", ".join([treatment_label, 'Delta_spo0A', genes_keep_unique_spo0a[g], str(delta_n_unique_spo0a[g]), str(round(P_values_unique_spo0a[g], 5))]))

    sys.stdout.write("Saving divergent genes between taxa......\n")
    output_file_bw_taxa.close()






def get_unique_diverged_genes_between_treatments_old():

    with open(pt.get_path()+'/data/divergence_skellam_between_treatments.pickle', 'rb') as handle:
        divergence_dict = pickle.load(handle)

    treatment_pairs_sets = set([frozenset(k) for k in treatment_pairs])

    output_file_bw_treatment = open(pt.get_path() + "/data/divergence_skellam_between_treatments.txt", "w")
    output_file_bw_treatment.write(", ".join(["Transfer regime comparison", "Transfer regime", "Strain", "Locus tag", "delta_n", "-log10[P]"]))

    for taxon in pt.taxa:

        # just pick the first treatment


        for treatment_pair_idx, treatment_pair in enumerate(treatment_pairs):

            treatment_pair_set = (treatment_pair[0], treatment_pair[1])

            delta_n = divergence_dict[taxon][treatment_pair_set]['delta_n']
            genes_keep = divergence_dict[taxon][treatment_pair_set]['genes_keep']
            P_values = divergence_dict[taxon][treatment_pair_set]['P_values']

            delta_n_1 = delta_n[delta_n>0]
            genes_keep_1 = genes_keep[delta_n>0]
            P_values_1 = P_values[delta_n>0]

            delta_n_2 = delta_n[delta_n<0]
            genes_keep_2= genes_keep[delta_n<0]
            P_values_2 = P_values[delta_n<0]

            genes_keep_set_1 = set(genes_keep_1)
            genes_keep_set_2 = set(genes_keep_2)

            for taxon_loop in pt.taxa:

                for treatment_pair_loop_idx, treatment_pair_loop in enumerate(treatment_pairs):

                    # skip the one you're looking at
                    if (taxon_loop == taxon) and (treatment_pair_loop == treatment_pair):
                        continue

                    treatment_pair_set_loop = (treatment_pair_loop[0], treatment_pair_loop[1])

                    delta_n_loop = divergence_dict[taxon_loop][treatment_pair_set_loop]['delta_n']
                    genes_keep_loop = divergence_dict[taxon_loop][treatment_pair_set_loop]['genes_keep']
                    P_values_loop = divergence_dict[taxon_loop][treatment_pair_set_loop]['P_values']

                    delta_n_loop_1 = delta_n_loop[delta_n_loop>0]
                    genes_keep_loop_1 = genes_keep_loop[delta_n_loop>0]
                    P_values_loop_1 = P_values_loop[delta_n_loop>0]

                    delta_n_loop_2 = delta_n_loop[delta_n_loop<0]
                    genes_keep_loop_2 = genes_keep_loop[delta_n_loop<0]
                    P_values_loop_2 = P_values_loop[delta_n_loop<0]

                    genes_keep_set_1 -= set(genes_keep_loop_1)
                    genes_keep_set_1 -= set(genes_keep_loop_2)

                    genes_keep_set_2 -= set(genes_keep_loop_1)
                    genes_keep_set_2 -= set(genes_keep_loop_2)


            treatment_comparison_label = "%d-day vs. %d-day" % (10**int(treatment_pair[0]), 10**int(treatment_pair[1]))
            treatment_1_label = "%d-day" % 10**int(treatment_pair[0])
            treatment_2_label = "%d-day" % 10**int(treatment_pair[1])

            if taxon == 'B':
                taxon_label = 'WT'
            else:
                taxon_label = 'Delta_spo0A'

            genes_keep_unique_1 = np.asarray(list(genes_keep_set_1))
            genes_keep_unique_idx_1 = [np.where(genes_keep_1 == k)[0][0] for k in genes_keep_unique_1]
            genes_keep_unique_idx_1 = np.asarray(genes_keep_unique_idx_1)
            if len(genes_keep_unique_idx_1) > 0 :

                genes_keep_unique_1 = genes_keep_1[genes_keep_unique_idx_1]
                delta_n_unique_1 = delta_n_1[genes_keep_unique_idx_1]
                P_values_unique_1 = P_values_1[genes_keep_unique_idx_1]

                delta_n_unique_1 = np.absolute(delta_n_unique_1)
                P_values_unique_1 = np.log10(P_values_unique_1)*-1

                for g in range(len(genes_keep_unique_1)):

                    output_file_bw_treatment.write("\n")
                    output_file_bw_treatment.write(", ".join([treatment_comparison_label, treatment_1_label, taxon_label, genes_keep_unique_1[g], str(delta_n_unique_1[g]), str(round(P_values_unique_1[g], 5))]))


            genes_keep_unique_2 = np.asarray(list(genes_keep_set_2))
            genes_keep_unique_idx_2 = [np.where(genes_keep_2 == k)[0][0] for k in genes_keep_unique_2]
            genes_keep_unique_idx_2 = np.asarray(genes_keep_unique_idx_2)
            if len(genes_keep_unique_idx_2) > 0 :

                genes_keep_unique_2 = genes_keep_2[genes_keep_unique_idx_2]
                delta_n_unique_2 = delta_n_2[genes_keep_unique_idx_2]
                P_values_unique_2 = P_values_2[genes_keep_unique_idx_2]

                delta_n_unique_2 = np.absolute(delta_n_unique_2)
                P_values_unique_2 = np.log10(P_values_unique_2)*-1

                for g in range(len(genes_keep_unique_2)):

                    output_file_bw_treatment.write("\n")
                    output_file_bw_treatment.write(", ".join([treatment_comparison_label, treatment_2_label, taxon_label, genes_keep_unique_2[g], str(delta_n_unique_2[g]), str(round(P_values_unique_2[g], 5))]))





            # go through all treats and


    output_file_bw_treatment.close()






def get_unique_diverged_genes_between_taxa():

    with open(pt.get_path()+'/data/divergence_skellam_between_taxa.pickle', 'rb') as handle:
        divergence_dict_between_taxa = pickle.load(handle)

    treatments_set = set(pt.treatments)

    output_file_bw_taxa = open(pt.get_path() + "/data/divergence_skellam_between_taxa.txt", "w")
    output_file_bw_taxa.write(", ".join(["Transfer regime", "Strain", "Locus tag", "delta_n", "-log10[P]"]))

    for treatment in divergence_dict_between_taxa.keys():
        other_treatments = treatments_set - set(treatment)

        delta_n = divergence_dict_between_taxa[treatment]['delta_n']
        genes_keep = divergence_dict_between_taxa[treatment]['genes_keep']
        P_values = divergence_dict_between_taxa[treatment]['P_values']

        delta_n_wt = delta_n[delta_n>0]
        genes_keep_wt = genes_keep[delta_n>0]
        P_values_wt = P_values[delta_n>0]

        delta_n_spo0a = delta_n[delta_n<0]
        genes_keep_spo0a = genes_keep[delta_n<0]
        P_values_spo0a = P_values[delta_n<0]

        genes_keep_set_wt = set(genes_keep_wt)
        genes_keep_set_spo0a = set(genes_keep_spo0a)

        treatment_label = '%d-day' % 10**int(treatment)

        for other_treatment in other_treatments:

            delta_n_other_treatment = divergence_dict_between_taxa[other_treatment]['delta_n']
            genes_keep_other_treatment = divergence_dict_between_taxa[other_treatment]['genes_keep']
            P_values_other_treatment = divergence_dict_between_taxa[other_treatment]['P_values']

            delta_n_other_treatment_wt = delta_n_other_treatment[delta_n_other_treatment>0]
            genes_keep_other_treatment_wt = genes_keep_other_treatment[delta_n_other_treatment>0]
            P_values_other_treatment_wt = P_values_other_treatment[delta_n_other_treatment>0]

            delta_n_other_treatment_spo0a = delta_n_other_treatment[delta_n_other_treatment<0]
            genes_keep_other_treatment_spo0a = genes_keep_other_treatment[delta_n_other_treatment<0]
            P_values_other_treatment_spo0a = P_values_other_treatment[delta_n_other_treatment<0]

            genes_keep_set_wt -= set(genes_keep_other_treatment_wt)
            genes_keep_set_wt -= set(genes_keep_other_treatment_spo0a)

            genes_keep_set_spo0a -= set(genes_keep_other_treatment_spo0a)
            genes_keep_set_spo0a -= set(genes_keep_other_treatment_wt)


        genes_keep_unique_wt = np.asarray(list(genes_keep_set_wt))
        genes_keep_unique_idx_wt = [np.where(genes_keep_wt == k)[0][0] for k in genes_keep_unique_wt]
        genes_keep_unique_idx_wt = np.asarray(genes_keep_unique_idx_wt)
        if len(genes_keep_unique_idx_wt) > 0 :

            genes_keep_unique_wt = genes_keep_wt[genes_keep_unique_idx_wt]
            delta_n_unique_wt = delta_n_wt[genes_keep_unique_idx_wt]
            P_values_unique_wt = P_values_wt[genes_keep_unique_idx_wt]

            delta_n_unique_wt = np.absolute(delta_n_unique_wt)
            P_values_unique_wt = np.log10(P_values_unique_wt)*-1

            print(treatment, 'WT', len(genes_keep_unique_wt))

            for g in range(len(genes_keep_unique_wt)):

                output_file_bw_taxa.write("\n")
                output_file_bw_taxa.write(", ".join([treatment_label, 'WT', genes_keep_unique_wt[g], str(delta_n_unique_wt[g]), str(round(P_values_unique_wt[g], 5))]))


        genes_keep_unique_spo0a = np.asarray(list(genes_keep_set_spo0a))
        genes_keep_unique_idx_spo0a = [np.where(genes_keep_spo0a == k)[0][0] for k in genes_keep_unique_spo0a]
        genes_keep_unique_idx_spo0a = np.asarray(genes_keep_unique_idx_spo0a)


        if len(genes_keep_unique_idx_spo0a) > 0 :

            genes_keep_unique_spo0a = genes_keep_spo0a[genes_keep_unique_idx_spo0a]
            delta_n_unique_spo0a = delta_n_spo0a[genes_keep_unique_idx_spo0a]
            P_values_unique_spo0a = P_values_spo0a[genes_keep_unique_idx_spo0a]

            delta_n_unique_spo0a = np.absolute(delta_n_unique_spo0a)
            P_values_unique_spo0a = np.log10(P_values_unique_spo0a)*-1

            print(treatment, 'spo0a', len(genes_keep_unique_spo0a))

            for g in range(len(genes_keep_unique_spo0a)):

                output_file_bw_taxa.write("\n")
                output_file_bw_taxa.write(", ".join([treatment_label, 'Delta_spo0A', genes_keep_unique_spo0a[g], str(delta_n_unique_spo0a[g]), str(round(P_values_unique_spo0a[g], 5))]))

    sys.stdout.write("Saving divergent genes between taxa......\n")
    output_file_bw_taxa.close()






get_unique_diverged_genes_between_taxa()
#calculate_between_taxa()
#calculate_between_treatments()
#get_unique_diverged_genes_between_taxa()
#get_unique_diverged_genes_between_treatments()
