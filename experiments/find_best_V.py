'''Search for the value of Voc that most closely finds a target optimum H-bar design.

Oliver Nakano-Baker
Jan. 2021'''

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append('..')
from gridgraph.utils import param_loader, start_logging
from gridgraph.numerical_circle_cell import simulate_H, simulate_iso


TARGET_PITCH = 0.15
TARGET_WIDTH = 0.027


def find_best_V(params):
    Vs = np.arange(.01, 1.01, .01)
    results = np.zeros((len(Vs), 2))
    for i, V in enumerate(Vs):
        inputs = {**params, 'Voc': V}
        _, w, b = simulate_H(inputs)
        results[i, :] = [w, b]

    errors = np.array([[TARGET_WIDTH, TARGET_PITCH]]) - results
    MSE = np.mean(np.square(errors), axis=1)
    plt.plot(Vs, MSE)
    plt.show()

    print(f'best V: {Vs[np.argmin(MSE)]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='../recipes/center_circle.csv',
                        help='CSV containing run parameters.')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)

    V = find_best_V({**vars(args), **params})
