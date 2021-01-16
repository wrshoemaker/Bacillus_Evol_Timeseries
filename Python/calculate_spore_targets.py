from __future__ import division
import numpy
import random
import os
import sys
import parse_file
import mutation_spectrum_utils
import timecourse_utils
import phylo_tools as pt

import matplotlib.pyplot as plt

n_samples = 10000

spore_locus_tags = parse_file.get_spore_locus_tags()

gene_data = parse_file.parse_gene_list('B')

gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data


pseudocount=1

for treatment_idx, treatment in enumerate(pt.treatments):
    fraction_list = []
    N_list = []
    for taxon in pt.taxa:
        significant_multiplicity_taxon_path = pt.get_path() + '/data/timecourse_final/parallel_genes_%s.txt' % (treatment+taxon)
        significant_multiplicity_taxon = open(significant_multiplicity_taxon_path, "r")
        enriched_genes = []
        enriched_spore_genes = []
        for i, line in enumerate( significant_multiplicity_taxon ):
            if i == 0:
                continue
            line = line.strip()
            items = line.split(",")
            enriched_genes.append(items[0])
            if items[0] in spore_locus_tags:
                enriched_spore_genes.append(items[0])



        N_all = len(enriched_genes)
        N_spore = len(enriched_spore_genes)
        fraction_spore_genes = (N_spore+pseudocount)/(N_all+pseudocount)

        fraction_list.append(fraction_spore_genes)
        N_list.append(N_all)


    delta_f = fraction_list[0] - fraction_list[1]

    delta_f_null = []
    for i in range(n_samples):

        sample_genes_B = random.sample(gene_names, N_list[0])
        sample_genes_S = random.sample(gene_names, N_list[1])

        sample_genes_spores_B = [x for x in sample_genes_B if x in spore_locus_tags]
        sample_genes_spores_S = [x for x in sample_genes_S if x in spore_locus_tags]

        f_sample_B = (len(sample_genes_spores_B)+pseudocount) / (N_list[0]+pseudocount)
        f_sample_S = (len(sample_genes_spores_S)+pseudocount) / (N_list[1]+pseudocount)

        delta_f_sample = f_sample_B - f_sample_S
        delta_f_null.append(delta_f_sample)

    delta_f_null = numpy.asarray(delta_f_null)

    if delta_f> numpy.median(delta_f_null):
        P = len(delta_f_null[delta_f_null >= delta_f]) / n_samples
    else:
        P = len(delta_f_null[delta_f_null < delta_f]) / n_samples
    #P = P/2

    standard_delta_f = (delta_f -  numpy.mean(delta_f_null)) / numpy.std( delta_f_null)


    sys.stdout.write("B vs. S %s-day delta fract= %.3f, std.delta fract= %.3f, P=%.3f\n" % (treatment, delta_f, standard_delta_f, P))
