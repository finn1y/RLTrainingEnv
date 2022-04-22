#!/usr/bin/env python3

import sys, os
import argparse, pickle
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

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

    plt.rc("font", size=28)
    fig = plt.figure(figsize=(16, 10))

    ax = plt.axes()
    ax.set_xlabel("Episode")
    ax.set_ylabel("avg. Reward across 5 episodes")
    ax.set_title(r'Comparison of $\epsilon$-min parameter for Q-Learning')
    ax.grid()

#    axins = inset_axes(ax, 6, 3, loc="lower right")
#    x1, y1, x2, y2 = 70, 0.33, 98, 0.412
#    axins.set_xlim(x1, x2)
#    axins.set_ylim(y1, y2)
#    axins.set_xticklabels([])
#    axins.set_yticklabels([])

    for directory in args.directory:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)

        if os.path.isdir(path):
            data = load_data(f'{path}/data.pkl')
            algorithm = directory.split('/')[-2]

            if algorithm == "qlearning":
                label = "Q-Learning"
            elif algorithm == "policy_gradient":
                label = "Policy Gradient"
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
            else:
                label = algorithm

            plt_data = [data[f'{args.data_type}'][i] for i in range(len(data[f'{args.data_type}']))]
            e = [i * 5 for i in range(20)]
            ax.plot(e, plt_data, label=label, linewidth=4)
            #axins.plot(e, plt_data, label=label, linewidth=4)


    ax.plot(avg, linewidth=4)

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title=r'$\epsilon$-min')

    #mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

    plt.draw()
    plt.savefig("./plot.png", bbox_inches="tight")

