from __future__ import division
import numpy
import sys
import parse_file
import mutation_spectrum_utils
import timecourse_utils
import phylo_tools as pt

import matplotlib.pyplot as plt


if len(sys.argv) > 1:
    level = sys.argv[1]
else:
    level = 'gene'

taxa = [ 'B', 'S']
treatments=pt.treatments
#treatments = ['2']
replicates = pt.replicates


nonsynonymous_types = set(['missense','nonsense'])
synonymous_types = set(['synonymous'])

non_appeared = {}
non_fixed = {}

syn_appeared = {}
syn_fixed = {}


non_appeared_spore = {}
non_fixed_spore = {}

syn_appeared_spore = {}
syn_fixed_spore = {}


#targeted_Lsyn = {}
#targeted_Lnon = {}
#targeted_fixed_Lsyn = {}
#targeted_fixed_Lnon = {}


gene_data = parse_file.parse_gene_list('B')

gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data

# to get the common gene names for each ID
gene_name_dict = dict(zip(gene_names, genes ))
protein_id_dict = dict(zip(gene_names, protein_ids ))

spore_locus_tags = parse_file.get_spore_locus_tags()
Lsyn_spore, Lnon_spore = parse_file.calculate_target_sizes_spore_genes()


Lsyn, Lnon, substitution_specific_synonymous_fraction = parse_file.calculate_synonymous_nonsynonymous_target_sizes('B')
position_gene_map, effective_gene_lengths, substitution_specific_synonymous_fraction = parse_file.create_annotation_map('B', gene_data=None)


#print([effective_gene_lengths[x]['synonymous'] for x in effective_gene_lengths.keys() if x in spore_protein_ids])

Lsyn = Lsyn - Lsyn_spore
Lnon = Lnon - Lnon_spore

populations = []

for taxon in taxa:
    for treatment in treatments:
        for replicate in replicates:

            population = treatment + taxon + replicate
            populations.append(population)

            non_appeared[population] = 1
            non_fixed[population] = 1

            syn_appeared[population] = 1
            syn_fixed[population] = 1

            non_appeared_spore[population] = 1
            non_fixed_spore[population] = 1

            syn_appeared_spore[population] = 1
            syn_fixed_spore[population] = 1

            #targeted_Lsyn[population] = 1
            #targeted_Lnon[population] = 1
            #targeted_fixed_Lsyn[population] = 1
            #targeted_fixed_Lnon[population] = 1

            mutations, depth_tuple = parse_file.parse_annotated_timecourse(population)
            population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths = depth_tuple
            state_times, state_trajectories = parse_file.parse_well_mixed_state_timecourse(population)

            num_processed_mutations = 0

            for mutation_idx in range(0,len(mutations)):

                location, gene_name, allele, var_type, codon, position_in_codon, AAs_count, test_statistic, pvalue, cutoff_idx, depth_fold_change, depth_change_pvalue, times, alts, depths, clone_times, clone_alts, clone_depths = mutations[mutation_idx]

                state_Ls = state_trajectories[mutation_idx]

                good_idxs, filtered_alts, filtered_depths = timecourse_utils.mask_timepoints(times, alts, depths, var_type, cutoff_idx, depth_fold_change, depth_change_pvalue)

                freqs = timecourse_utils.estimate_frequencies(filtered_alts, filtered_depths)

                masked_times = times[good_idxs]
                masked_freqs = freqs[good_idxs]
                masked_state_Ls = state_Ls[good_idxs]

                fixed_weight = timecourse_utils.calculate_fixed_weight(masked_state_Ls[-1],masked_freqs[-1])

                #if var_type in nonsynonymous_types or var_type in synonymous_types:
                #    targeted_Lnon[population] += (1-substitution_specific_synonymous_fraction[allele])
                #    targeted_fixed_Lnon[population] += fixed_weight*(1-substitution_specific_synonymous_fraction[allele])
                #    targeted_Lsyn[population] += substitution_specific_synonymous_fraction[allele]
                #    targeted_fixed_Lsyn[population] += fixed_weight*substitution_specific_synonymous_fraction[allele]

                if var_type in nonsynonymous_types:

                    if gene_name in spore_locus_tags:
                        non_appeared_spore[population] += 1
                        non_fixed_spore[population] += fixed_weight

                    else:
                        non_appeared[population]+=1
                        non_fixed[population]+=fixed_weight


                elif var_type in synonymous_types:

                    if gene_name in spore_locus_tags:
                        syn_appeared_spore[population] += 1
                        syn_fixed_spore[population] += fixed_weight

                    else:
                        syn_appeared[population]+=1
                        syn_fixed[population]+=fixed_weight


                num_processed_mutations+=1




total_non_appeared = sum([non_appeared[population] for population in populations])
total_non_fixed = sum([non_fixed[population] for population in populations])
total_syn_appeared = sum([syn_appeared[population] for population in populations])
total_syn_fixed = sum([syn_fixed[population] for population in populations])



print(syn_fixed_spore)
print(syn_appeared_spore)

total_non_appeared_spore = sum([non_appeared_spore[population] for population in populations])
total_non_fixed_spore = sum([non_fixed_spore[population] for population in populations])
total_syn_appeared_spore = sum([syn_appeared_spore[population] for population in populations])
total_syn_fixed_spore = sum([syn_fixed_spore[population] for population in populations])


dnds_appeared = total_non_appeared/total_syn_appeared*Lsyn/Lnon
dnds_fixed = total_non_fixed/total_syn_fixed*Lsyn/Lnon
dnds_appeared_spore = total_non_appeared_spore/total_syn_appeared_spore*Lsyn_spore/Lnon_spore
dnds_fixed_spore = total_non_fixed_spore/total_syn_fixed_spore*Lsyn_spore/Lnon_spore

#print(dnds_appeared_spore, dnds_fixed_spore)








# plot dN/dS
fig, ax = plt.subplots(figsize=(4,4))

for treatment in ['0', '1']:

    for taxon in taxa:

        populations = [ '%s%s%s' % (treatment, taxon, replicate) for replicate in replicates ]

        taxon_treatment_dnds_appeared = [non_appeared[population]/(syn_appeared[population]+(syn_appeared[population]==0))*Lsyn/Lnon for population in populations]
        taxon_treatment_dnds_fixed = [non_fixed[population]/(syn_fixed[population]+(syn_fixed[population]==0))*Lsyn/Lnon for population in populations]

        taxon_treatment_dnds_appeared_spore = [non_appeared_spore[population]/(syn_appeared_spore[population]+(syn_appeared_spore[population]==0))*Lsyn_spore/Lnon_spore for population in populations]
        taxon_treatment_dnds_fixed_spore = [non_fixed_spore[population]/(syn_fixed_spore[population]+(syn_fixed_spore[population]==0))*Lsyn_spore/Lnon_spore for population in populations]

        #print(non_fixed_spore)

        taxon_treatment_dnds_appeared = numpy.asarray(taxon_treatment_dnds_appeared)
        taxon_treatment_dnds_fixed = numpy.asarray(taxon_treatment_dnds_fixed)

        taxon_treatment_dnds_appeared_spore = numpy.asarray(taxon_treatment_dnds_appeared_spore)
        taxon_treatment_dnds_fixed_spore = numpy.asarray(taxon_treatment_dnds_fixed_spore)


        print(taxon_treatment_dnds_appeared)
        print(taxon_treatment_dnds_appeared)


        alpha = 1  - taxon_treatment_dnds_appeared/taxon_treatment_dnds_fixed
        alpha_spore = 1  - taxon_treatment_dnds_appeared_spore/taxon_treatment_dnds_fixed_spore


        ax.scatter(alpha, alpha_spore, s = 180, \
            linewidth=3, facecolors=pt.get_scatter_facecolor(taxon, treatment), \
            edgecolors=pt.get_colors(treatment), marker=pt.plot_species_marker(taxon), \
            alpha=0.6, zorder=2)


    #t, p = stats.ttest_ind(taxon_treatment_dnds_appeared_B, taxon_treatment_dnds_appeared_S, equal_var=True)
    #print(treatment, t, p)


fig.subplots_adjust(hspace=0.3, wspace=0.5)
fig_name = pt.get_path() + '/figs/dn_ds_spore.jpg'
fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
