import random
import copy


# normal mutation
def mutate(params_comb, params_lst, rate_of_mutation):
    for k in range(len(params_comb)):
        if random.random() < rate_of_mutation:
            params_selected_randomly = copy.deepcopy(params_lst[k][random.randint(0, len(params_lst[k]) - 1)])
            params_comb[k] = copy.deepcopy(params_selected_randomly)
    return params_comb


# rank-base mutation
def mutate_on_the_basis_of_rank(params_comb, params_lst, max_rate_of_mutation, i, population):
    rate_of_mutation = (i/population) * max_rate_of_mutation
    for k in range(len(params_comb)):
        if random.random() < rate_of_mutation:
            params_selected_randomly = copy.deepcopy(params_lst[k][random.randint(0, len(params_lst[k]) - 1)])
            params_comb[k] = copy.deepcopy(params_selected_randomly)
    return params_comb
