import copy
import random
import datetime

import numpy as np
import pandas as pd

from .mutate import mutate_normal, mutate_rank_base
from .terminate import terminate
from .make_active_lst import make_active_lst
from .get_info_about_params import get_max_params, \
    get_max_fitness, get_min_fitness, get_mean_fitness, get_median_fitness, get_sd_fitness

"""
Hyper-parameters need to be explained:
    - params: search_range (explained below)
    - objective: objective function
    - p_m [float]: mutation rate (default: 0.1=10%)
    - generation [int]: generation size (default: 30)
    - population [int]: population size (population: 100)
    - rank_based_mutation [True or False]: 
        whether you use rank-based mutation or normal mutation (default: True)
        - if use rank-based mutation, p_m means max mutation rate
        - need information, please refer to documents
    - early_stopping [True or False]: whether you use early_stopping or not (default: False)
        - if no improve of best_fitness and mean_fitness, terminate.
    - maximizing [True or False]: maximizing or minimizing problem 
    - preserving_calc [True or False]: preserving calculation or not
    - history [2, 1, or 0]: outputs the detail or less (default: 0)
        - 2: all information;   best_params, best_fitness, 
                                best_fitness_lst, worst_fitness_lst, 
                                mean_fitness_lst, median_fitness_lst, sd_fitness_lst, 
                                search_history_lst
        - 1: best_params, best_fitness
        - 0: best_params only
    - verbose [2, 1, or 0]: print the detail or less at each step (default: 0)
        - 2: detail
        - 1: less
        - 0: nothing
    - seed [int]: seed at randomizing

Params range should be specified as follows.

    from sseopt import search_space
    params = {
        'x1': search_space.categorical(['relu', 'tanh']), # list(candidates)
        'x2': search_space.discrete(-1.0, 1.0, 0.2), # min, max, step
        'x3': search_space.discrete_int(-4, 2), # min, max
        'x4': search_space.fixed(1) # a fixed value
    }



Hiroya MAKINO
Grad. School of Informatics, Nagoya University

ver1.0, Aug. 27 2020
ver1.1. Nov. 18 2020 | not make file, but return list at visualize function.
ver2.0. Jan. 24 2021 | you can choose categorical- or discrete-coded. more sophisticated coding.
"""


class params_comb:
    pass


class common_schema:
    pass
    # x: [1,2,3]
    # y: [1, 3]


class SSEOpt:
    
    def __init__(
            self, 
            params, objective, p_m=0.10, generation=30, population=100,
            rank_based_mutation=True, early_stopping=False, maximizing=True, preserving_calc=True, 
            history=0, verbose=0, seed=168):
        if params is None:
            TypeError("You must specify the params range")
        self.params = params
        self.keys = [key for key in params.keys()]
        for key in self.keys:
            setattr(params_comb, key, "")
            setattr(common_schema, key, [])
            
        if objective is None:
            TypeError("You must specify the objective function")
        self.objective = objective
        
        self.rate_of_mutation = p_m
        self.num_of_gens = generation
        self.population = population

        self.rank_based_mutation = rank_based_mutation
        self.early_stopping = early_stopping
        self.maximizing = maximizing
        self.preserving_calc = preserving_calc
        self.history = history
        self.verbose = verbose

        random.seed(seed)
        
    def fit(self):
        current_lst = [[] for pop in range(self.population)]

        # ---------------------
        # initial population
        # ---------------------
        for i in range(self.population):
            temp_params_comb = params_comb()
            for key in self.keys:
                setattr(temp_params_comb, key, self.params[key].select())
            current_lst[i] = temp_params_comb
        next_lst = copy.deepcopy(current_lst)
        
        best_params_lst = []
        best_fitness_lst = []
        worst_fitness_lst = []
        mean_fitness_lst = []
        median_fitness_lst = []
        sd_fitness_lst = []
        search_history_lst = []
        
        calc_fitnesses_lst = []
        calc_params_combs = []

        for gen in range(self.num_of_gens):
            print('\n')
            print('*** generation', gen, "/", self.num_of_gens, " *************")

            current_lst = copy.deepcopy(next_lst)
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
                              format('    calculating', i + 1, '/', self.population), end="")
                        
                    if self.preserving_calc:
                        if len(calc_params_combs) > 0:
                            for calc_params_comb_index in range(len(calc_params_combs)):
                                if self.is_params_comb_same(calc_params_combs[calc_params_comb_index], 
                                                            current_lst[i]):
                                    current_fitness_lst[i] = calc_fitnesses_lst[calc_params_comb_index]
                                    break
                                elif calc_params_comb_index == len(calc_params_combs) -1:
                                    score = self.output_fitness(current_lst[i])
                                    current_fitness_lst[i] = score
                                    calc_fitnesses_lst.append(score)
                                    calc_params_combs.append(copy.deepcopy(current_lst[i]))
                        else:   
                            score = self.output_fitness(current_lst[0])
                            current_fitness_lst[0] = score
                            calc_fitnesses_lst.append(score)
                            calc_params_combs.append(copy.deepcopy(current_lst[0]))
                    else:
                        current_fitness_lst[i] = self.output_fitness(current_lst[i])

                # ---------------------
                # information
                # ---------------------
                best_params_lst += [get_max_params(current_lst, current_fitness_lst, self.keys)]
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
            else:
                # ---------------------
                # calculate fitness
                # ---------------------
                for i in range(self.population):
                    if self.verbose > 0:
                        print("\r{0} {1} {2} {3} ".
                              format('    calculating', i + 1, '/', self.population), end="")
                        
                    if self.preserving_calc:
                        if len(calc_params_combs) > 0:
                            for calc_params_comb_index in range(len(calc_params_combs)):
                                if self.is_params_comb_same(calc_params_combs[calc_params_comb_index], 
                                                            current_lst[i]):
                                    current_fitness_lst[i] = calc_fitnesses_lst[calc_params_comb_index]
                                    break
                                elif calc_params_comb_index == len(calc_params_combs) -1:
                                    score = -self.output_fitness(current_lst[i])
                                    current_fitness_lst[i] = score
                                    calc_fitnesses_lst.append(score)
                                    calc_params_combs.append(copy.deepcopy(current_lst[i]))
                        else:   
                            score = -self.output_fitness(current_lst[0])
                            current_fitness_lst[0] = score
                            calc_fitnesses_lst.append(score)
                            calc_params_combs.append(copy.deepcopy(current_lst[0]))
                    else:
                        current_fitness_lst[i] = -self.output_fitness(current_lst[i])
                        
                # ---------------------
                # information
                # ---------------------
                best_params_lst += [get_max_params(current_lst, current_fitness_lst, self.keys)]
                best_fitness_lst += [-get_max_fitness(current_fitness_lst)]
                worst_fitness_lst += [-get_min_fitness(current_fitness_lst)]
                mean_fitness_lst += [-get_mean_fitness(current_fitness_lst)]
                median_fitness_lst += [-get_median_fitness(current_fitness_lst)]
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

            active_lst_index = copy.deepcopy(make_active_lst(self.population, sorted_index, sorted_current_fitness))

            # ---------------------
            # extract schema
            # ---------------------
            current_lst_copy = copy.deepcopy(current_lst)
            next_schemata_lst = []
            for l in range(self.population):
                schemata_lst = copy.deepcopy([current_lst_copy[active_lst_index[l][m]] 
                                              for m in range(len(active_lst_index[l]))])
                next_schemata_lst.append(copy.deepcopy(self.extract_schema(schemata_lst)))

            search_history_lst.append(copy.deepcopy(next_schemata_lst))

            for i in range(self.population):
                # ---------------------
                # make individuals by random selection
                # ---------------------
                next_lst[i] = self.make_individual(next_schemata_lst[i])

                # ---------------------
                # mutation
                # ---------------------
                if self.rank_based_mutation:
                    next_lst[i] = copy.deepcopy(
                        mutate_rank_base(next_lst[i], self.params, self.keys, self.rate_of_mutation, i, self.population))
                else:
                    next_lst[i] = copy.deepcopy(
                        mutate_normal(next_lst[i], self.params, self.keys, self.rate_of_mutation))
                
        max_index = best_fitness_lst.index(max(best_fitness_lst))

        if self.history == 2:
            return best_params_lst[max_index], best_fitness_lst[max_index],\
                   best_fitness_lst, worst_fitness_lst, mean_fitness_lst, median_fitness_lst, sd_fitness_lst,\
                   search_history_lst
        elif self.history == 1:
            return best_params_lst[max_index], best_fitness_lst[max_index]
        elif self.history == 0:
            return best_params_lst[max_index]

    def output_fitness(self, params_comb_temp):
        fitness = self.objective(params_comb_temp)
        return fitness
        
    def extract_schema(self, params_combs_lst):
        common_schema_temp = common_schema()
        for key in self.keys:
            params_lst = []
            for p in params_combs_lst:
                params_lst.append(getattr(p, key))
            params_set = list(set(params_lst))
            setattr(common_schema_temp, key, params_set)
        return common_schema_temp
    
    def make_individual(self, common_schemata):
        individual = params_comb()
        for key in self.keys:
            params_lst = getattr(common_schemata, key)
            setattr(individual, key, random.choice(params_lst))
        return individual
    
    def is_params_comb_same(self, a, b):
        result = True
        for key in self.keys:
            if getattr(a, key) != getattr(b, key):
                result = False
                break
        return result
       
    def visualize(self, search_history_lst):
        params_dic_key_lst=[]
        params_dic_value_lst=[]
        
        for key in self.keys:
            value_temp = self.params[key].full_candidates_lst()
            params_dic_value_lst.append(value_temp)
            params_dic_key_lst += [key] * len(value_temp)
        
        history_visualized_lst = [[0] * len(sum(params_dic_value_lst, [])) for i in range(self.num_of_gens)]

        for gen in range(self.num_of_gens):
            chromosome_serial_num = 0
            for chromosome_attentioned in range(len(params_dic_value_lst)):
                for num_in_chromosome in range(len(params_dic_value_lst[chromosome_attentioned])):
                    for pop in range(self.population):
                        if params_dic_value_lst[chromosome_attentioned][num_in_chromosome] in \
                                getattr(search_history_lst[gen][pop], self.keys[chromosome_attentioned]):
                            history_visualized_lst[gen][chromosome_serial_num] \
                                += 1 / len(getattr(search_history_lst[gen][pop], self.keys[chromosome_attentioned]))
                    chromosome_serial_num += 1

        history_visualized_array = np.array(history_visualized_lst, dtype='float32')
        
        for i in range(self.num_of_gens):
            history_visualized_array[i] = history_visualized_array[i] / self.population
        history_visualized_lst = history_visualized_array.tolist()
        history_visualized_lst = [params_dic_key_lst] + [sum(params_dic_value_lst, [])] + history_visualized_lst

        return history_visualized_lst
