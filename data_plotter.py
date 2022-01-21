#!/usr/bin/env python3

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

    parser.add_argument("-e", "--environments", choices=envs, default=envs, nargs="+", help="Environment data was gathered on")
    parser.add_argument("-a", "--algorithms", choices=algorithms, default=algorithms, nargs="+", help="RL Algorithm data was gathered using")
    parser.add_argument("-d", "--data-type", default="rewards", nargs="+", help="types of data to be plotted")

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
    #list of all possible environements
    envs = ["maze-random-5x5-v0", "maze-random-10x10-v0", "maze-random-100x100-v0", 
            "maze-sample-5x5-v0", "maze-sample-10x10-v0", "maze-sample-100x100-v0", "gym_robot_maze:robot-maze-v0", 
            "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"]
    #list of all possible algorithms
    algorithms = ["qlearning", "dqn", "drqn", "policy_gradient", "actor_critic", "ddpg", "ma_actor_critic"]

    args = get_args(envs, algorithms)

    for a in args.algorithms:
        a_path = os.path.join(os.getcwd(), "saved_data", args.environments[0], a)

        if os.path.isdir(a_path):
            dir_list = os.listdir(a_path)

            for dir_name in dir_list:
                if "training" in dir_name:
                    number = int(dir_name[-1])
                    data = load_data(f'{a_path}/{dir_name}/data.pkl')

                    avg_reward = [np.average(data[f'{args.data_type}'][i]) for i in range(len(data[f'{args.data_type}']))]
                    plt.plot(avg_reward, label=f'{a}{number}')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

