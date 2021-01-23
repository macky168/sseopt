from copy import deepcopy
import numpy as np

"""
old version

def make_active_lst(population, original_sorted_index, sorted_current_fitness):
    temp_sorted_index = [i for i in range(population)]
    active_lst = [[temp_sorted_index[0]]]

    # ---------------------
    # SPECIAL OCCASION
    # if best = worst, active_lst is [original_sorted_index, original_sorted_index, ..., original_sorted_index]
    # ---------------------
    if round(sorted_current_fitness[0], 4) == round(sorted_current_fitness[-1], 4):
        active_lst = [0] * population
        for i in range(population):
            active_lst[i] = original_sorted_index

    # ---------------------
    # else
    # ---------------------
    else:
        i = 0
        while i < population-1:
            print("i", i)
            key_index = deepcopy(active_lst[i][-1]) + 1
            next_active_lst_a = deepcopy(active_lst[i])
            next_active_lst_b = deepcopy(active_lst[i])
            next_active_lst_a.append(temp_sorted_index[key_index])
            next_active_lst_b = next_active_lst_b[:-1]
            next_active_lst_b.append(temp_sorted_index[key_index])
            active_lst.append(next_active_lst_a)
            active_lst.append(next_active_lst_b)

            avg_fitness = [None] * len(active_lst)
            for j in range(len(active_lst)):
                avg_fitness[j] = np.average(np.array([sorted_current_fitness[active_lst[j][k]] for k in range(len(active_lst[j]))]))

            sorted_index = np.argsort(avg_fitness)[::-1]
            print(sorted_index)

            active_lst_tmp = [active_lst[sorted_index[l]] for l in range(len(sorted_index))]
            if len(active_lst_tmp) > population:
                active_lst = deepcopy(active_lst_tmp[:population])
            else:
                active_lst = deepcopy(active_lst_tmp)

            i += 1

        for k in range(population):
            for j in range(len(active_lst[k])):
                active_lst[k][j] = original_sorted_index[active_lst[k][j]]

    return active_lst
"""


"""
new version is below
This version doesn't use sort, but extract the best subset at each step.
"""


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


def main():
    """
    import random
    population = 1000
    sorted_index = [i for i in range(1000)]
    sorted_current_fitness = [random.randint(0, 1000) for i in range(1000)]
    print("sorted_index", sorted_index)
    active_lst = make_active_lst(population, sorted_index, sorted_current_fitness)
    print(active_lst)
    print("len", len(active_lst))
    """

    population = 9
    sorted_index = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    sorted_current_fitness = [10000, 9999, 800, 61, 60, 50, 40, 30, 20]
    print(make_active_lst(population, sorted_index, sorted_current_fitness))


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    process_time = time.time() - start
    print(process_time)
