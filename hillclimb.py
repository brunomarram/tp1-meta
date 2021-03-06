from datetime import datetime

import math
import numpy as np
import matplotlib.pyplot as plt

from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from random import seed

seed(datetime.now())


def objective_func1(v):
    x, y = v
    return math.sin(x+y) + pow(x-y, 2) - 1.5*x + 2.5*y + 1


def objective_func2(v):
    x, y = v
    return -(y + 47) * math.sin(math.sqrt((abs(x/2 + (y + 47))))) - x * math.sin(math.sqrt((abs(x - (y + 47)))))


def in_bounds(point, bounds):
    # enumerate all dimensions of the point
    for d in range(len(bounds)):
        # check if out of bounds for this dimension
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True


def hillclimbing(objective, bounds, n_iterations, step_size):
    # store the initial point
    solution = bounds[:, 0] + rand(len(bounds)) * \
        (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    solution_eval = objective(solution)
    # run the hill climb
    for i in range(n_iterations):
        # take a step
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = solution + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidte_eval = objective(candidate)
        # check if we should keep the new point
        if candidte_eval <= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidte_eval
    return [solution, solution_eval]


# define range for input
bounds = asarray([[511, 512], [404, 405]])
scores = []

for i in range(0, 30):
    # define the total iterations
    n_iterations = 10000
    # define the maximum step size
    step_size = 0.2
    # perform the hill climbing search
    best, score = hillclimbing(
        objective_func2, bounds, n_iterations, step_size)

    print('Done!')
    print('f(%s) = %f' % (best, score))
    scores.append(score)

print(f"Mínimo: {min(scores)}")
print(f"Máximo: {max(scores)}")
print(f"Média: {np.mean(scores)}")
print(f"Desvio Padrão: {np.std(scores)}")

plt.boxplot(scores)
plt.show()
