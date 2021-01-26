import random
import copy


# normal mutation
def mutate_normal(params_comb_temp, params, keys, rate_of_mutation):
    for key in keys:
        if random.random() < rate_of_mutation:
            setattr(params_comb_temp, key, params[key].mutate(getattr(params_comb_temp, key)))
        else:
            pass
            # no mutation
    return params_comb_temp


# rank-base mutation
def mutate_rank_base(params_comb_temp, params, keys, max_rate_of_mutation, i, population):
    rate_of_mutation = (i/population) * max_rate_of_mutation
    for key in keys:
        if random.random() < rate_of_mutation:
            setattr(params_comb_temp, key, params[key].mutate(getattr(params_comb_temp, key)))
        else:
            pass
            # no mutation
    return params_comb_temp
