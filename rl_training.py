#!/usr/bin/env python3

import os, sys
import argparse, pickle
import gym
import gym_maze
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from algorithms.qlearning import QLearning
from algorithms.dqn import DQN
from algorithms.policy_grad import PolicyGradient
from algorithms.actor_critic import ActorCritic
from algorithms.ddpg import DDPG

def get_args(envs, algorithms):
    """
        function to get the command line arguments

        returns a namespace of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("Environment", type=str, choices=envs, help="Environment to train agent on")
    parser.add_argument("Algorithm", type=str, choices=algorithms, help="RL Algorithm to train agent with")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("-t", "--time-steps", type=int, default=10000, help="Number of time steps per episode")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Number of batches sampled from replay memory during training")
    parser.add_argument("-r", "--render", action="store_true", help="Flag to render environment")
    parser.add_argument("-m", "--model-path", default=None, help="Path to the saved model to continue training")
    parser.add_argument("-s", "--hidden-size", type=int, default=128, help="Number of neurons in the hidden layer of neural nets")
    parser.add_argument("-p", "--plot", action="store_true", help="Flag to plot data after completion")

    return parser.parse_args()

def save_data(data, agent, env, algorithm):
    """
        function to save data in a pickle file gathered during training in the saved_data directory

        data is a dictionary of the data to be saved

        agent is a reference of the agent trained on the environment with the algorithm

        env is the name of the environment the data was gathered on

        algorithm is the name of the RL algorithm used to gather the data
    """
    path = os.path.join(os.getcwd(), "saved_data", env, algorithm)

    #make directory if not already exists
    if not os.path.isdir(path):
        os.makedirs(path)

    dir_list = os.listdir(path)
    number = 1

    #number data files incrementally to prevent overwriting old data
    for file_name in dir_list:
        if "data" in file_name:
            number = int(file_name[-5]) + 1

    with open(f'{path}/data{number}.pkl', "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if algorithm != "qlearning":
        agent.save_model(f'{path}/model{number}')

if __name__ == "__main__":
    #list of all possible environements
    envs = ["maze-random-5x5-v0", "maze-random-10x10-v0", "maze-random-100x100-v0", 
            "maze-sample-5x5-v0", "maze-sample-10x10-v0", "maze-sample-100x100-v0", "gym_robot_maze:robot-maze-v0", 
            "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"]
    #list of all possible algorithms
    algorithms = ["qlearning", "dqn", "drqn", "policy_gradient", "actor_critic", "ddpg"]

    args = get_args(envs, algorithms)

    if args.Algorithm == "ddpg" and args.Environment in envs[:9]:
        raise Exception(f'DDPG can only be simulated with continuous action spaces: {envs[9:]}')
    if args.Algorithm in algorithms[:5] and args.Environment in envs[9:]:
        raise Exception(f'{args.Algorithm} can only be simulated with discrete action spaces: {envs[:9]}')

    if args.Environment in envs[:6]:
        env = gym.make(args.Environment, enable_render=args.render)
    elif args.Environment in envs[6:7]:
        env = gym.make(args.Environment, is_render=args.render)
    else:
        env = gym.make(args.Environment)

    if type(env.action_space) == gym.spaces.discrete.Discrete:
        actions = [i for i in range(env.action_space.n)]
#        actions = [i for i in range(env.action_space.start, env.action_space.start + env.action_space.n)]
        actions = [actions, actions]
    elif type(env.action_space) == gym.spaces.box.Box:
        actions = [env.action_space.low, env.action_space.high]

    if type(env.observation_space) == gym.spaces.discrete.Discrete:
        observations = [i for i in range(env.observation_space.n)]
#        observations = [i for i in range(env.observation_space.start, env.observation_space.start + env.observation_space.n)]
        observations = [observations, observations]
    elif type(env.observation_space) == gym.spaces.box.Box:
        observations = [env.observation_space.low, env.observation_space.high]

    batch_size = args.batch_size
    
    if args.Algorithm == "qlearning":
        agent = QLearning([observations, actions]) 

    if args.Algorithm in algorithms[1:3]:
        recurrent = True if args.Algorithm == "drqn" else False
        agent = DQN([observations, args.hidden_size, actions], lr_decay_steps=args.time_steps,  DRQN=recurrent, saved_path=args.model_path)

    if args.Algorithm == "policy_gradient":
        agent = PolicyGradient([observations, args.hidden_size, actions], lr_decay_steps=args.time_steps, saved_path=args.model_path)

    if args.Algorithm == "actor_critic": 
        agent = ActorCritic([observations, args.hidden_size, actions], lr_decay_steps=args.time_steps, saved_path=args.model_path)

    if args.Algorithm == "ddpg":
        agent = DDPG([observations, args.hidden_size, actions], lr_decay_steps=args.time_steps, saved_path=args.model_path)
    
    successes = 0
    all_losses = []
    rewards = []

    if args.render:
        env.render()
    
    for i in range(args.episodes):
        obv = env.reset()
        ep_losses = []
        total_reward = 0
        done = False

        for t in range(args.time_steps):
            if args.render:
                env.render()

            action = agent.get_action(obv)
            next_obv, reward, done, _ = env.step(action)

            if args.Algorithm == "qlearning":
                agent.train(obv, action, reward, next_obv)
                
            if args.Algorithm in algorithms[1:6]:
                agent.store_step(obv, action, reward, next_obv)

            obv = next_obv
            total_reward += reward

            if (args.Algorithm in algorithms[1:3] or args.Algorithm == "ddpg") and t >= batch_size and t % 4 == 0:
                loss = agent.train(batch_size)
                ep_losses.append(loss)
                
                if t % 20 == 0:
                    agent.update_target_net()

            if done or t == (args.time_steps - 1):
                print(f'Episode {i} finished after {t} time steps with total reward = {total_reward}')

                if args.Algorithm in algorithms[:3]:
                    agent.update_parameters(i, args.episodes)
                
                if done:
                    successes += 1

                if args.Algorithm in algorithms[3:5]:
                    loss = agent.train()
                    ep_losses.append(loss)

                rewards.append(total_reward)
                all_losses.append(ep_losses)
                break

            if args.Environment in envs[:6] and env.is_game_over():
                sys.exit(0)

    print(f'Training complete with {successes}/{args.episodes} episodes completed')

    data = {"Parameters": agent.get_parameters(), "rewards": rewards, "losses": all_losses, "successes": successes}
    save_data(data, agent, args.Environment, args.Algorithm)

    if args.plot:
        plt.plot(data["rewards"], color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

    sys.exit(0)


