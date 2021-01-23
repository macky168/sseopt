import numpy as np


def get_max_params(current_lst, current_fitness_lst):
    max_index = current_fitness_lst.index(max(current_fitness_lst))
    return current_lst[max_index]


def get_max_fitness(current_fitness_lst):
    return max(current_fitness_lst)


def get_min_fitness(current_fitness_lst):
    return min(current_fitness_lst)


def get_mean_fitness(current_fitness_lst):
    return np.mean(np.array(current_fitness_lst))


def get_median_fitness(current_fitness_lst):
    return np.median(np.array(current_fitness_lst))


def get_sd_fitness(current_fitness_lst):
    return np.std(np.array(current_fitness_lst))
