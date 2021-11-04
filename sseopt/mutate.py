import random
import copy

CONST = 0.0

# ---------------------
# mutate individual
# ---------------------
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
def mutate_rank_base(params_comb_temp, params, keys, max_rate_of_mutation, i, i_worst):
    rate_of_mutation = (i/i_worst) * max_rate_of_mutation + CONST
    for key in keys:
        if random.random() < rate_of_mutation:
            setattr(params_comb_temp, key, params[key].mutate(getattr(params_comb_temp, key)))
        else:
            pass
            # no mutation
    return params_comb_temp

# ---------------------
# mutate each bit
# ---------------------
# normal mutation
def mutate_normal_each_key(params_comb_temp, params, key, rate_of_mutation):
    if random.random() < rate_of_mutation:
        setattr(params_comb_temp, key, params[key].mutate(getattr(params_comb_temp, key)))
    else:
        pass
        # no mutation
    return params_comb_temp

# rank-base mutation
def mutate_rank_base_each_key(params_comb_temp, params, key, max_rate_of_mutation, i, i_worst):
    rate_of_mutation = (i/i_worst) * max_rate_of_mutation + CONST
    if random.random() < rate_of_mutation:
        setattr(params_comb_temp, key, params[key].mutate(getattr(params_comb_temp, key)))
    else:
        pass
        # no mutation
    return params_comb_temp