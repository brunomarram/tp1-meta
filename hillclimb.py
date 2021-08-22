from datetime import datetime

import math
import numpy as np
import matplotlib.pyplot as plt

from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from random import seed

seed(datetime.now())


def objective_func1(x, y):
    return math.sin(x+y) + pow(x-y, 2) - 1.5*x + 2.5*y + 1


def objective_func2(x, y):
    return math.sin(x+y) + pow(x-y, 2) - 1.5*x + 2.5*y + 1


def hillclimbing(objective, bounds_x, bounds_y, n_iterations, step_size):
    # generate an initial point
    x = bounds_x[:, 0] + rand(len(bounds_x)) * \
        (bounds_x[:, 1] - bounds_x[:, 0])
    y = bounds_y[:, 0] + rand(len(bounds_y)) * \
        (bounds_y[:, 1] - bounds_y[:, 0])
    # evaluate the initial point
    solution_eval = objective(x, y)
    # run the hill climb
    for i in range(n_iterations):
        # take a step
        candidate_x = x + randn(len(bounds_x)) * step_size
        candidate_y = y + randn(len(bounds_y)) * step_size
        # evaluate candidate point
        candidate_eval = objective(candidate_x, candidate_y)
        # check if we should keep the new point
        if candidate_eval <= solution_eval:
            # store the new point
            solution_x, solution_y, solution_eval = candidate_x, candidate_y, candidate_eval
            # report progress
            # print('>%d f(%s, %s) = %.5f' %
            #       (i, solution_x, solution_y, solution_eval))
    return [solution_x, solution_y, solution_eval]


# define range for input
bounds_x = asarray([[-1.5, 4]])
bounds_y = asarray([[-3, 4]])
scores = []

for i in range(0, 30):
    # define the total iterations
    n_iterations = 5000
    # define the maximum step size
    step_size = 0.1
    # perform the hill climbing search
    best_x, best_y, score = hillclimbing(
        objective_func1, bounds_x, bounds_y, n_iterations, step_size)

    print('Done!')
    print('f(%s, %s) = %f' % (best_x, best_y, score))
    scores.append(score[0])

print(f"Mínimo: {min(scores)}")
print(f"Máximo: {max(scores)}")
print(f"Média: {np.mean(scores)}")
print(f"Desvio Padrão: {np.std(scores)}")

plt.boxplot(scores)
plt.show()
