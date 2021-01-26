from copy import deepcopy

import numpy as np


def make_active_lst(population, original_sorted_index, sorted_current_fitness):

    temp_sorted_index = [i for i in range(population)]    # index of index
    subset_lst = [[temp_sorted_index[0]]]
    active_lst = []
    active_lst_avg_fitness = []

    # ---------------------
    # SPECIAL OCCASION
    # if best = worst, active_lst is [original_sorted_index, original_sorted_index, ..., original_sorted_index]
    # ---------------------
    if round(sorted_current_fitness[0], 4) == round(sorted_current_fitness[-1], 4):
        subset_lst = [0] * population
        for i in range(population):
            subset_lst[i] = original_sorted_index

    # ---------------------
    # else
    # ---------------------
    else:
        i = 0
        while i < population - 1:
            key_index = deepcopy(subset_lst[i][-1]) + 1
            next_active_lst_a = deepcopy(subset_lst[i])
            next_active_lst_b = deepcopy(subset_lst[i])
            next_active_lst_a.append(temp_sorted_index[key_index])
            next_active_lst_b = next_active_lst_b[:-1]
            next_active_lst_b.append(temp_sorted_index[key_index])

            active_lst.append(next_active_lst_a)
            active_lst.append(next_active_lst_b)

            next_active_lst_a_fitness = np.average(
                np.array([sorted_current_fitness[next_active_lst_a[k]] for k in range(len(next_active_lst_a))]))
            next_active_lst_b_fitness = np.average(
                np.array([sorted_current_fitness[next_active_lst_b[k]] for k in range(len(next_active_lst_b))]))

            active_lst_avg_fitness.append(next_active_lst_a_fitness)
            active_lst_avg_fitness.append(next_active_lst_b_fitness)

            max_index = int(np.argmax(active_lst_avg_fitness))
            subset_lst.append(active_lst[max_index])

            active_lst.pop(max_index)
            active_lst_avg_fitness.pop(max_index)

            i += 1

        for k in range(population):
            for j in range(len(subset_lst[k])):
                subset_lst[k][j] = original_sorted_index[subset_lst[k][j]]

    return subset_lst
