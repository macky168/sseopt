import sseopt
from sseopt import search_space

params_range = {
    'x1': search_space.categorical(['relu', 'tanh']), # list(candidates)
    'x2': search_space.discrete(-1.0, 1.0, 0.2), # min, max, step
    'x3': search_space.discrete_int(-4, 2), # min, max
    'x4': search_space.fixed(1) # a fixed value
}

def objective(params):
    return params.x2 ** 2

def main():
    instance = sseopt.SSEOpt(params_range, objective=objective)
    best_params = instance.fit() 
    print(best_params)
    
if __name__ == "__main__":
    main()