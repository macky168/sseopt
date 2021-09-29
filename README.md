# sseopt
Stochastic Schemata Exploiter-based hyper-parameter optimization of Machine Learning

## Description
Stochastic Schemata Exploiter (SSE) is one of the Evolutionary Algorithms.
SSE-based hyper-parameter optimization method, called SSEopt, has interesting features: quick convergence, the small number of control parameters, and the process visualization.
SSEopt is particularly good at large combinatorial optimization problems in discrete or categorical space.

## Requirement
- numpy 1.19.2
- tensorflow 2.3.0
- pandas 1.2.1

## Usage

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
   

## Licence
[MIT](https://github.com/macky168/sseopt/blob/main/LICENCE.txt)

## Reference
[1] A. N. Aizawa, “Evolving SSE: A stochastic schemata exploiter,” in Proceedings of the first IEEE conference on evolutionary computation. IEEE world congress on computational intelligence, 1994, pp. 525–529.

[2] H. Makino, X. Feng, and E. Kita, “Stochastic schemata exploiter-based optimization of convolutional neural network,” in IEEE international conference on systems, man, and cybernetics, 2020, pp. 4365–4371.

## Author
Hiroya MAKINO (makino.hiroya@e.mbox.nagoya-u.ac.jp)

Grad. School of Informatics, Nagoya University, Japan
