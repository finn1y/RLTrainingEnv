#!/usr/bin/env python

import sys, os
import argparse, pickle

import numpy as np
import matplotlib.pyplot as plt

def get_args(envs, algorithms):
    """
        function to get the command line arguments

        returns a namespace of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("directory", type=str, nargs='+', help="path to directory where training data is saved")
    parser.add_argument("-d", "--data-type", default="rewards", nargs="+", help="type of data to be plotted")

    return parser.parse_args()

def load_data(path):
    """
        function to load data from a pickle file

        path is the path the pickel file
    """
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
    else:
        raise FileNotFoundError

    return data

if __name__ == "__main__":
    args = get_args()

    for directory in args.directory:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)

        if os.path.isdir(path):
            data = load_data(f'{path}/data.pkl')

            avg_data = [np.average(data[f'{args.data_type}'][i]) for i in range(len(data[f'{args.data_type}']))]
            plt.plot(avg_data, label=f'{os.path.basename(directory)}')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

