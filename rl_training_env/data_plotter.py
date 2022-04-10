#!/usr/bin/env python3

import sys, os
import argparse, pickle

import numpy as np
import matplotlib.pyplot as plt

def get_args():
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

    env = args.directory[0].split('/')[-3]

    for directory in args.directory:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)

        if os.path.isdir(path):
            data = load_data(f'{path}/data.pkl')
            algorithm = directory.split('/')[-2]

            if algorithm == "qlearning":
                label = "Q-Learning"
            elif algorithm == "policy_grad":
                label = "REINFORCE"
            elif algorithm == "dqn":
                label = "DQN"
            elif algorithm == "drqn":
                label = "DRQN"
            elif algorithm == "ddpg":
                label = "DDPG"
            elif algorithm == "actor_critic":
                label = "Actor Critic"
            elif algorithm == "ddrqn":
                label = "DDRQN"
            elif algorithm == "ma_actor_critic":
                label = "Multi-Agent Actor Critic"

            plt_data = [data[f'{args.data_type}'][i] for i in range(len(data[f'{args.data_type}']))]
            plt_data = [np.average(plt_data[i:i+5]) for i in range(0, len(plt_data), 5)]
            e = [i * 5 for i in range(100)]
            plt.plot(e, plt_data, label=label)

    plt.title(f'Comparison of Algorithms on {env} Environment')
    plt.xlabel("Episode")
    plt.ylabel("Avg. Reward Across 10 Episodes")
    plt.legend()
    plt.show()

