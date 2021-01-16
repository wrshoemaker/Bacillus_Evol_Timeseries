from __future__ import division
import numpy
import random
import sys
import pickle
import copy

from itertools import groupby

import phylo_tools as pt
from itertools import combinations

import scipy.stats as stats
import  matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




numpy.random.seed(123456789)



def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def run_simulation(M = 10, c = 0.00001):
    #4,292,969
    # mutation rate from Lynch paper, assume 10% sites are beneficial
    #mu = (3.28*10**-10 ) * 0.1
    #L =  4292969
    # keep order of magnitude for conveinance
    mu = (1.0*10**-10 )
    L =  1000000

    N = 10**6
    #M = 10
    K = N/M

    s_scale = 10**-2
    # scale = beta = 1/lambda
    # expected value of fitness effect
    # average time in a dormant state = M
    # average time in a dormant state = 1/(n_dormant_to_active/M ) = M/c*K*M = 1/(c*K) = M/(c*N)
    n_active_to_dormant = int(c*N)
    n_dormant_to_active = int(c*K*M)

    if n_active_to_dormant != n_dormant_to_active:
        print("Unqueal number of individuals switching states!!")

    # rate of entering dormancy, per-capita = c
    # rate of exiting dormancy, per-capita = c*K
    #d = (c* K) / N
    #r = c / M

    # double mutants in a single simulation slow the simulation so we're assuming single mutants
    # e.g., the largest lineage size = 10**6, generated L*mu*N (~1000) mutants per-generation
    # probability that an individual gets two mutations ~= 10 ** -7

    sampled_timepoints = {}
    # offset counts starting at zero
    generations = 3300 + 1
    #generations = 991
    generations_to_sample = [330*i for i in range(1, 11)]
    #generations_to_sample = [330*i for i in range(1, 3)]
    n_clone_lineages = 0

    clone_size_dict = {}
    clone_size_dict[n_clone_lineages] = {}
    clone_size_dict[n_clone_lineages]['n_clone_active'] = N
    clone_size_dict[n_clone_lineages]['n_clone_dormant'] = M
    clone_size_dict[n_clone_lineages]['s'] = 1
    clone_size_dict[n_clone_lineages]['mutations'] = set([])

    # pre-assign fitness benefits to all sites
    all_sites = set(range(L))
    fitness_effects = numpy.random.exponential(scale=s_scale, size=L)

    # dict of what clones have a given mutation
    for generation in range(1, generations+1):
        # generate dormancy transition rates for all lineages
        # get keys and make sure they're in the same order
        #clones_active = [ clone_i for clone_i in clone_size_dict.keys() if ('n_clone_active' in clone_size_dict[clone_i]) and (clone_size_dict[clone_i]['n_clone_active'] > 0) ]
        #clones_active.sort()
        #clones_dormant = [ clone_i for clone_i in clone_size_dict.keys() if ('n_clone_dormant' in clone_size_dict[clone_i]) and (clone_size_dict[clone_i]['n_clone_dormant'] > 0)  ]
        #clones_dormant.sort()

        # get array of clone labels, the number of times each label is in the array is the size of the lineage
        clone_labels_active = [[int(clone_i)] * clone_size_dict[clone_i]['n_clone_active'] for clone_i in clone_size_dict.keys()]
        clone_labels_dormant = [[int(clone_i)] * clone_size_dict[clone_i]['n_clone_dormant'] for clone_i in clone_size_dict.keys() if ('n_clone_dormant' in clone_size_dict[clone_i]) and (clone_size_dict[clone_i]['n_clone_dormant'] > 0 )]

        clone_labels_active = numpy.concatenate(clone_labels_active).ravel()
        clone_labels_dormant = numpy.concatenate(clone_labels_dormant).ravel()

        clone_labels_active = clone_labels_active.astype(numpy.int)
        clone_labels_active = clone_labels_active.astype(numpy.int)


        # number of dormant individuals not constant???
        active_to_dormant_sample = numpy.random.choice(clone_labels_active, size = n_active_to_dormant, replace=False)
        active_to_dormant_sample_bincount = numpy.bincount(active_to_dormant_sample)
        active_to_dormant_sample_bincount_nonzero = numpy.nonzero(active_to_dormant_sample_bincount)[0]

        dormant_to_active_sample = numpy.random.choice(clone_labels_dormant, size = n_dormant_to_active, replace=False)
        dormant_to_active_sample_bincount = numpy.bincount(dormant_to_active_sample)
        dormant_to_active_sample_bincount_nonzero = numpy.nonzero(dormant_to_active_sample_bincount)[0]

        for active_to_dormant_clone_i, active_to_dormant_n_clone_i in zip(active_to_dormant_sample_bincount_nonzero, active_to_dormant_sample_bincount[active_to_dormant_sample_bincount_nonzero]):

            clone_size_dict[active_to_dormant_clone_i]['n_clone_active'] -= active_to_dormant_n_clone_i

            if 'n_clone_dormant' not in clone_size_dict[active_to_dormant_clone_i]:
                clone_size_dict[active_to_dormant_clone_i]['n_clone_dormant'] = 0

            clone_size_dict[active_to_dormant_clone_i]['n_clone_dormant'] += active_to_dormant_n_clone_i


        for dormant_to_active_clone_i, dormant_to_active_n_clone_i in zip(dormant_to_active_sample_bincount_nonzero, dormant_to_active_sample_bincount[dormant_to_active_sample_bincount_nonzero]):

            clone_size_dict[dormant_to_active_clone_i]['n_clone_dormant'] -= dormant_to_active_n_clone_i

            if 'n_clone_dormant' not in clone_size_dict[dormant_to_active_clone_i]:
                clone_size_dict[dormant_to_active_clone_i]['n_clone_active'] = 0

            clone_size_dict[dormant_to_active_clone_i]['n_clone_active'] += dormant_to_active_n_clone_i

        # now move on to evolution
        for clone_i in list(clone_size_dict):

            if (clone_size_dict[clone_i]['n_clone_dormant'] == 0):

                if (clone_size_dict[clone_i]['n_clone_active'] == 0):
                    del clone_size_dict[clone_i]
                    continue

                else:
                    continue

            n_clone_i = clone_size_dict[clone_i]['n_clone_active']

            # mutation step#
            # lineage size can't be negative
            n_mutations_clone = min(numpy.random.poisson(mu*L*n_clone_i), n_clone_i)
            if n_mutations_clone == 0:
                continue
            # remove these individuals from the clone
            clone_size_dict[clone_i]['n_clone_active'] -= n_mutations_clone
            # all individuals in the clone have the same mutations
            # so just sample from nonmutated sites in the ancestral clone
            non_mutated_sites = all_sites - clone_size_dict[clone_i]['mutations']

            # sample without replacement
            #mutated_sites = random.sample(non_mutated_sites, n_mutations_clone)
            mutated_sites = numpy.random.choice(list(non_mutated_sites), size=n_mutations_clone, replace=False)
            #unique, counts = numpy.unique(mutated_sites, return_counts=True)
            for mutated_site in mutated_sites:

                n_clone_lineages += 1

                clone_size_dict[n_clone_lineages] = {}
                clone_size_dict[n_clone_lineages]['n_clone_active'] = 1
                clone_size_dict[n_clone_lineages]['n_clone_dormant'] = 0
                clone_size_dict[n_clone_lineages]['s'] = clone_size_dict[clone_i]['s'] + fitness_effects[mutated_site]
                clone_size_dict[n_clone_lineages]['mutations'] = clone_size_dict[clone_i]['mutations'].copy()
                clone_size_dict[n_clone_lineages]['mutations'].add(mutated_site)

            #if (clone_size_dict[clone_i]['n_clone_active'] == 0) and (clone_size_dict[clone_i]['n_clone_dormant'] == 0):
            #    del clone_size_dict[clone_i]


        #sampling_numerator = numpy.asarray( [ clone_size_dict[clone_i]['n_clone']*numpy.exp(clone_size_dict[clone_i]['s']) for clone_i in sorted(clone_size_dict.keys())] )
        sampling_numerator = numpy.asarray( [ clone_size_dict[clone_i]['n_clone_active']*numpy.exp(clone_size_dict[clone_i]['s']) for clone_i in clone_size_dict.keys()] )
        sampling_probability = sampling_numerator / sum(sampling_numerator)
        clone_sizes_after_selection = numpy.random.multinomial(N, sampling_probability)

        for clone_i_idx, clone_i in enumerate(list(clone_size_dict)):
            clone_i_size = clone_sizes_after_selection[clone_i_idx]

            #if clone_i_size == 0:
            #    del clone_size_dict[clone_i]
            #else:
            clone_size_dict[clone_i]['n_clone_active'] = clone_i_size


        if generation %100 == 0:

            sys.stderr.write("%d generations...\n" % generation)


        if generation in generations_to_sample:
            #clone_size_dict_copy = clone_size_dict.copy()
            clone_size_dict_copy = copy.deepcopy(clone_size_dict)
            sampled_timepoints[generation] = clone_size_dict_copy



        N = sum( [ clone_size_dict[x]['n_clone_active'] for x in  clone_size_dict.keys() ] )
        M = sum( [ clone_size_dict[x]['n_clone_dormant'] for x in  clone_size_dict.keys() ] )

    #saved_data_file='%s/data/simulations/test2.dat' % (pt.get_path())

    return sampled_timepoints
    #with open(saved_data_file, 'wb') as outfile:
    #    pickle.dump(sampled_timepoints, outfile, protocol=pickle.HIGHEST_PROTOCOL)




def parse_simulation_output(sampled_timepoints):

    #saved_data_file='%s/data/simulations/test2.dat' % (pt.get_path())
    #sampled_timepoints = pickle.load( open(saved_data_file, "rb" ) )

    allele_freq_trajectory_dict = {}

    for generation, generation_dict in sampled_timepoints.items():

        N = sum( [ generation_dict[x]['n_clone_active'] for x in  generation_dict.keys() ] )
        M = sum( [ generation_dict[x]['n_clone_dormant'] for x in  generation_dict.keys() ] )
        all_individuals = N+M

        for genotype_,  genotype_clone_dict_ in generation_dict.items():

            #if (genotype_clone_dict_['n_clone_active']>0) or (genotype_clone_dict_['n_clone_dormant']>0):
            #    print(genotype_clone_dict_['n_clone_active'], genotype_clone_dict_['n_clone_dormant'] )

            for mutation in genotype_clone_dict_['mutations']:

                if mutation not in allele_freq_trajectory_dict:
                    allele_freq_trajectory_dict[mutation] = {}
                allele_freq_trajectory_dict[mutation][generation] = 0
                allele_freq_trajectory_dict[mutation][generation] += genotype_clone_dict_['n_clone_active']
                allele_freq_trajectory_dict[mutation][generation] += genotype_clone_dict_['n_clone_dormant']

        #print(allele_freq_trajectory_dict)
        for mutation in allele_freq_trajectory_dict.keys():
            if generation in allele_freq_trajectory_dict[mutation]:
                allele_freq_trajectory_dict[mutation][generation] = allele_freq_trajectory_dict[mutation][generation]/all_individuals

    # now go through and get statistics
    # delta f
    delta_f = []
    ratio_f = []
    r2_f = []
    max_f = []
    for mutation, time_dict in allele_freq_trajectory_dict.items():
        time_points = list(time_dict.keys())
        if len(time_points) <2:
            continue

        max_f.append(max(time_dict.values()))

        time_points.sort()
        for t_idx in range(0, len(time_points)-1 ):
            f_t_delta = time_dict[time_points[t_idx+1]]
            f_t = time_dict[time_points[t_idx]]
            # no fixation or extinction events
            if (f_t_delta>float(0)) and (f_t>float(0)) and (f_t_delta<float(1)) and (f_t<float(1)):
                delta_f_i = numpy.absolute(f_t_delta - f_t)
                if delta_f_i > float(0):
                    delta_f.append( numpy.absolute(f_t_delta - f_t) )
                ratio_f.append(f_t_delta/ f_t)

    all_mutation_pairs = combinations(allele_freq_trajectory_dict.keys(), 2)

    for mutation_pair in all_mutation_pairs:

        time_dict_i = allele_freq_trajectory_dict[mutation_pair[0]]
        time_dict_j = allele_freq_trajectory_dict[mutation_pair[1]]

        if (len(time_dict_i) == 1) or (len(time_dict_j) == 1):
            continue
        times_i = list(time_dict_i.keys())
        times_j = list(time_dict_j.keys())

        times_intersection = set(times_i) & set(times_j)
        if len(times_intersection) < 3:
            continue

        freqs_i = []
        freqs_j = []
        for time_ in times_intersection:
            freq_i_t = time_dict_i[time_]
            freq_j_t = time_dict_j[time_]

            if (freq_i_t>float(0)) and (freq_i_t<float(1)) and (freq_j_t>float(0)) and (freq_j_t<float(1)):
                freqs_i.append(freq_i_t)
                freqs_j.append(freq_j_t)

        if (all_equal(freqs_i) == True) or (all_equal(freqs_j) == True):
            continue

        if len(freqs_j)>=3:
            r2 = stats.pearsonr(freqs_i, freqs_j)[0] ** 2
            r2_f.append(r2)

    #delta_f = numpy.asarray(delta_f)
    #ratio_f = numpy.asarray(ratio_f)
    #r2_f = numpy.asarray(r2_f)

    return delta_f, ratio_f, r2_f, max_f



#parse_simulation_output()
M_list = numpy.logspace(1, 6, num=20, base=10.0)
#M_list = [10, 1000, 10000]
c_list = [0.00001]

def run_all_simulations():

    #K = N/M
    #n_active_to_dormant = int(c*N)
    #n_dormant_to_active = int(c*K*M)

    flatten = lambda t: [item for sublist in t for item in sublist]
    simulation_final_dict = {}
    for M in M_list:

        M = int(M)

        simulation_final_dict[M] = {}

        for c in c_list:

            simulation_final_dict[M][c] = {}

            delta_f_list = []
            ratio_f_list = []
            r2_f_list = []
            max_f_list = []

            for i in range(10):

                sys.stderr.write("M=%d, c=%f, Simulation %d\n" % (M, c, i))

                sampled_timepoints = run_simulation(M=M, c=c)

                delta_f, ratio_f, r2_f, max_f = parse_simulation_output(sampled_timepoints)

                delta_f_list.append(delta_f)
                ratio_f_list.append(ratio_f)
                r2_f_list.append(r2_f)
                max_f_list.append(max_f)


            #delta_f_pooled = numpy.concatenate(delta_f_list)
            #ratio_f_pooled = numpy.concatenate(ratio_f_list)
            #r2_f_pooled = numpy.concatenate(r2_f_list)

            delta_f_pooled = flatten(delta_f_list)
            ratio_f_pooled = flatten(ratio_f_list)
            r2_f_pooled = flatten(r2_f_list)
            max_f_pooled = flatten(max_f_list)

            simulation_final_dict[M][c]['delta_f'] = delta_f_pooled
            simulation_final_dict[M][c]['ratio_f'] = ratio_f_pooled
            simulation_final_dict[M][c]['r2_f'] = r2_f_pooled
            simulation_final_dict[M][c]['max_f'] = max_f_pooled

    sys.stderr.write("Simulations done! Saving pickle......\n")

    saved_data_file = '%s/data/simulations/all_seedbank_sizes.dat' % pt.get_path()
    with open(saved_data_file, 'wb') as outfile:
        pickle.dump(simulation_final_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)



run_all_simulations()



def plot_():
    saved_data_file='%s/data/simulations/all_seedbank_sizes.dat' % (pt.get_path())
    sampled_timepoints = pickle.load( open(saved_data_file, "rb" ) )

    metrics = ['delta_f', 'ratio_f', 'r2_f']

    fig = plt.figure(figsize = (10, 8))
    gs = gridspec.GridSpec(nrows=3, ncols=1)

    lines = ['-', '--', ':']

    for metric_idx, metric in enumerate(metrics):

        ax_i = fig.add_subplot(gs[metric_idx, 0])
        ax_i.set_yscale('log', base=10)

        if metric_idx != 2:
            ax_i.set_xscale('log', base=10)

        for M_idx, M in enumerate(M_list):

            f_max_array_sort = numpy.sort(sampled_timepoints[M][metric])

            cdf = 1-numpy.arange(len(f_max_array_sort))/float(len(f_max_array_sort))

            ax_i.plot(f_max_array_sort, cdf, c ='k', ls=lines[M_idx], lw=3, alpha=0.8)


    fig_name = pt.get_path() + '/figs/simulation.jpg'
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(fig_name, format='jpg', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()
