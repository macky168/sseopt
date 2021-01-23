import copy
import random
import numpy as np
import pandas as pd
import datetime

from .extract_schema import extract_schema
from .mutate import mutate, mutate_on_the_basis_of_rank

from .terminate import terminate
from .make_active_lst import make_active_lst
from .get_info_about_params import get_max_params, \
    get_max_fitness, get_min_fitness, get_mean_fitness, get_median_fitness, get_sd_fitness

"""
Parameters need to be explained:

- objective: objective function
- p_m [float]: mutation rate (default: 0.1=10%)
- generation [int]: generation size (default: 30)
- population [int]: population size (population: 100)
- rank_based_mutation [True or False]: whether you use rank-based mutation or normal mutation (default: True)
    - if use rank-based mutation, p_m means max mutation rate
    - need information, please refer to documents
- early_stopping [True or False]: whether you use early_stopping or not (default: False)
    - if no improve of best_fitness and mean_fitness, terminate.
- maximizing [True or False]: maximizing or minimizing problem 
- history [2, 1, or 0]: outputs the detail or less (default: 0)
    - 2: all information; best_params, best_fitness, best_fitness_lst, worst_fitness_lst, mean_fitness_lst, median_fitness_lst, sd_fitness_lst, search_history_lst
    - 1: best_params, best_fitness
    - 0: best_params only
- verbose [2, 1, or 0]: print the detail or less at each step (default: 0)
    - 2: detail
    - 1: less
    - 0: nothing
- seed [int]: seed at randomizing

Hiroya MAKINO
Grad. School of Informatics, Nagoya University

ver1.0, Aug. 27 2020
ver1.1. Nov. 18 2020 | not make file, but return list at visualize function.
"""


class SSEOpt:
    def __init__(self, params, objective, p_m=0.10, generation=30, population=100,
                 rank_based_mutation=True, early_stopping=False, maximizing=True, history=0, verbose=0, seed=168):
        self.params = [params]

        if objective is None:
            TypeError("You must specify objective function")
        self.objective = objective

        self.num_of_kinds_of_params = len(params)

        self.all_candidate_params = []
        for p in self.params:
            items = p.items()
            keys, values = zip(*items)
            self.keys = keys
            for v in values:
                self.all_candidate_params += [v]

        self.rate_of_mutation = p_m
        self.num_of_gens = generation
        self.population = population

        self.rank_based_mutation = rank_based_mutation
        self.early_stopping = early_stopping
        self.maximizing = maximizing
        self.history = history
        self.verbose = verbose

        random.seed(seed)

    def fit(self):
        current_lst = [[] for pop in range(self.population)]
        gene_len = self.num_of_kinds_of_params

        # ---------------------
        # produce initial group
        # ---------------------
        for i in range(self.population):
            temp_params_comb = [None] * gene_len
            for j in range(gene_len):
                temp_params_comb[j] \
                    = self.all_candidate_params[j][random.randint(0, len(self.all_candidate_params[j]) - 1)]
            current_lst[i] = temp_params_comb
        next_lst = current_lst

        best_params_lst = []
        best_fitness_lst = []
        worst_fitness_lst = []
        mean_fitness_lst = []
        median_fitness_lst = []
        sd_fitness_lst = []
        search_history_lst = []

        for gen in range(self.num_of_gens):
            print('\n')
            print('*** generation', gen, "/", self.num_of_gens, " ***")

            current_lst = next_lst
            current_fitness_lst = [None] * self.population
            next_lst = [0] * self.population

            # ---------------------
            # maximizing
            # ---------------------
            if self.maximizing:
                # ---------------------
                # calculate fitness
                # ---------------------
                for i in range(self.population):
                    if self.verbose > 0:
                        print("\r{0} {1} {2} {3} ".
                              format('  calculating', i + 1, '/', self.population), end="")
                    current_fitness_lst[i] = self.output_fitness(current_lst[i])

                # ---------------------
                # information
                # ---------------------
                best_params_lst += [get_max_params(current_lst, current_fitness_lst)]
                best_fitness_lst += [get_max_fitness(current_fitness_lst)]
                worst_fitness_lst += [get_min_fitness(current_fitness_lst)]
                mean_fitness_lst += [get_mean_fitness(current_fitness_lst)]
                median_fitness_lst += [get_median_fitness(current_fitness_lst)]
                sd_fitness_lst += [get_sd_fitness(current_fitness_lst)]

                if self.verbose > 1:
                    print('\n')
                    print('best parameters is : ', end="")
                    v_index = 0
                    for k in self.keys:
                        print(k, "", best_params_lst[-1][v_index], end=",   ")
                        v_index += 1
                    print('')
                    print('best fitness is : ', best_fitness_lst[-1])
                    print('worst fitness is : ', worst_fitness_lst[-1])
                    print('mean fitness is : ', mean_fitness_lst[-1])
                    print('median fitness is : ', median_fitness_lst[-1])
                    print('sd fitness is : ', sd_fitness_lst[-1])
                    print('\n')

            # ---------------------
            # minimizing
            # ---------------------
            elif not self.maximizing:
                # ---------------------
                # calculate fitness
                # ---------------------
                for i in range(self.population):
                    if self.verbose > 0:
                        print("\r{0} {1} {2} {3} ".
                              format('  calculating', i + 1, '/', self.population), end="")
                    current_fitness_lst[i] = -self.output_fitness(current_lst[i])

                # ---------------------
                # information
                # ---------------------
                best_params_lst += [get_max_params(current_lst, current_fitness_lst)]
                best_fitness_lst += [get_max_fitness(current_fitness_lst)]
                worst_fitness_lst += [get_min_fitness(current_fitness_lst)]
                mean_fitness_lst += [get_mean_fitness(current_fitness_lst)]
                median_fitness_lst += [get_median_fitness(current_fitness_lst)]
                sd_fitness_lst += [get_sd_fitness(current_fitness_lst)]

                if self.verbose > 1:
                    print('\n')
                    print('best parameters is : ', end="")
                    v_index = 0
                    for k in self.keys:
                        print(k, "", best_params_lst[-1][v_index], end=",   ")
                        v_index += 1
                    print('')
                    print('best fitness is : ', -best_fitness_lst[-1])
                    print('worst fitness is : ', -worst_fitness_lst[-1])
                    print('mean fitness is : ', -mean_fitness_lst[-1])
                    print('median fitness is : ', median_fitness_lst[-1])
                    print('sd fitness is : ', sd_fitness_lst[-1])
                    print('\n')

            # ---------------------
            # termination
            # ---------------------
            if self.early_stopping:
                if terminate(best_fitness_lst, mean_fitness_lst):
                    break

            # ---------------------
            # make individual subset
            # ---------------------
            index_lst \
                = [i for i in range(len(current_fitness_lst))]
            index_and_current_fitness \
                = np.concatenate((np.array(index_lst).reshape(-1, 1),
                                  np.array(current_fitness_lst).reshape(-1, 1)), axis=1)
            sorted_index_and_current_fitness \
                = sorted(index_and_current_fitness, key=lambda x: float(x[1]), reverse=True)
            sorted_index \
                = [0] * len(current_fitness_lst)
            sorted_current_fitness \
                = [0] * len(current_fitness_lst)
            for k in range(len(sorted_index_and_current_fitness)):
                sorted_index[k] = sorted_index_and_current_fitness[k][0].astype(int)
                sorted_current_fitness[k] = sorted_index_and_current_fitness[k][1]

            active_lst = copy.deepcopy(make_active_lst(self.population, sorted_index, sorted_current_fitness))

            # ---------------------
            # extract schema
            # ---------------------
            schema_lst = [[] for pop in range(self.population)]
            current_lst_copy = copy.deepcopy(current_lst)
            for l in range(self.population):
                schema_lst[l] = copy.deepcopy([current_lst_copy[active_lst[l][m]] for m in range(len(active_lst[l]))])
                next_lst[l] = copy.deepcopy(extract_schema(schema_lst[l]))

            search_history_lst.append(copy.deepcopy(next_lst))

            for i in range(self.population):
                # ---------------------
                # produce children by random selection
                # ---------------------
                temp_params_comb = [None] * gene_len
                for j in range(gene_len):
                    temp_params_comb[j] \
                        = copy.deepcopy(next_lst[i][j][random.randint(0, len(next_lst[i][j]) - 1)])
                next_lst[i] = copy.deepcopy(temp_params_comb)

                # ---------------------
                # mutate
                # ---------------------
                if self.rank_based_mutation:
                    next_lst[i] = copy.deepcopy(
                        mutate_on_the_basis_of_rank(next_lst[i], self.all_candidate_params, self.rate_of_mutation, i,
                                                    self.population))
                else:
                    next_lst[i] = copy.deepcopy(mutate(next_lst[i], self.all_candidate_params, self.rate_of_mutation))

        max_index = best_fitness_lst.index(max(best_fitness_lst))

        if self.history == 2:
            return best_params_lst[max_index], best_fitness_lst[
                max_index], best_fitness_lst, worst_fitness_lst, mean_fitness_lst, median_fitness_lst, sd_fitness_lst, search_history_lst
        elif self.history == 1:
            return best_params_lst[max_index], best_fitness_lst[max_index]
        elif self.history == 0:
            return best_params_lst[max_index]

    def output_fitness(self, params_comb):
        args = {key: val for key, val in zip(self.keys, params_comb)}
        fitness = self.objective(**args)
        return fitness

    def visualize(self, search_history_lst, params, dic_index_lst):
        population_range = [self.population]

        params_dic_lst = []

        dic_item_lst = []

        for k in dic_index_lst:
            params_dic_lst.append(params[k])
            for param in params[k]:
                dic_item_lst.append(str(k + '_' + str(param)))

        params_dic_lst_length = 0
        for item in params_dic_lst:
            params_dic_lst_length += len(item)

        for population in population_range:
            history_visualized_lst = [[0] * params_dic_lst_length for i in range(self.num_of_gens)]

            for gen in range(self.num_of_gens):
                chromosome_serial_num = 0
                for chromosome_attentioned in range(len(params_dic_lst)):
                    for num_in_chromosome in range(len(params_dic_lst[chromosome_attentioned])):
                        for pop in range(population):
                            if params_dic_lst[chromosome_attentioned][num_in_chromosome] in \
                                    search_history_lst[gen][pop][chromosome_attentioned]:
                                history_visualized_lst[gen][chromosome_serial_num] += 1 / len(
                                    search_history_lst[gen][pop][chromosome_attentioned])
                        chromosome_serial_num += 1

            history_visualized_array = np.array(history_visualized_lst, dtype='float32')

            for i in range(self.num_of_gens):
                history_visualized_array[i] = history_visualized_array[i] / population
            history_visualized_lst = history_visualized_array.tolist()
            history_visualized_lst = [dic_item_lst] + history_visualized_lst

            return history_visualized_lst
