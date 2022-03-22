#!/usr/bin/env python3

#python module to run a reinforcement learning algorithm on an openai gym environment with a variety of settable parameters

#-----------------------------------------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------------------------------------

import os, sys, subprocess
import argparse, pickle
import logging
import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt

from algorithms import *

#-----------------------------------------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------------------------------------

def get_args(envs: list, algorithms: list):
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
    parser.add_argument("-a", "--agents", type=int, default=1, help="Number of agents")
    parser.add_argument("-d", "--directory", type=str, default=None, help="Save the results from the training to the specified directory")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor to multiply by future expected rewards in the algorithm. Should be greater than 0 and less than 1")
    parser.add_argument("--epsilon-max", type=float, default=1.0, help="Epsilon max is the intial value of epsilon for epsilon-greedy policy. Should be greater than 0 and less than or equal to 1")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Epsilon min is the final value of epsilon for epsilon-greedy policy which is decayed to over training from epsilon max. Should be greater than 0 and less then epsilon max")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.0001, help="Learning rate of the algorithm. Should be greater than 0 and less than 1")
    parser.add_argument("--decay", type=float, default=0.999, help="Decay the base used for exponential decay of the learning rate (and epsilon where appropriate) during training if set to 1 will have no decay. Should be greater than 0 and less than 1")
    parser.add_argument("--maze-load-path", type=str, default=None, help="Path to load maze object from, for gym-robot-maze environment")
    parser.add_argument("--maze-save-path", type=str, default=None, help="Path to save maze object to, for gym-robot-maze environment")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity level")

    return parser.parse_args()

def save_data(path: str, data: dict, env: str, algorithm: str):
    """
        function to save data in a pickle file gathered during training in the directory at path
        directory has the following structure:

            -path
                -data.pkl

        path is the path to the directory to store the data in

        data is the data to be saved

        env is the name of the environment the data was gathered on

        algorithm is the name of the RL algorithm used to gather the data
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    #make directory if not already exists
    if not os.path.isdir(path):
        os.makedirs(path)

    #save parameters to pickle file
    with open(f'{path}/data.pkl', "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_installed_envs() -> list:
    """
        function to check which envs are installed on the system and return those available

        returns list of envs available on this system given installed packages
    """
    envs = []
    
    #run pip show [package] to see if package is installed
    gym = subprocess.check_output(["pip", "show", "gym"])
    maze = subprocess.check_output(["pip", "show", "gym-maze"])
    robot_maze = subprocess.check_output(["pip", "show", "gym-robot-maze"])

    #if package is installed then no warning is given so related envs can be added to list
    if not b'WARNING: Package(s) not found:' in gym:
        envs.extend(["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"])
    #if package is not installed then related envs cannot be added to list as not available on this system
    else:
        logging.warning("gym not installed, unable to use environments installed with gym package. Install with: pip install gym")

    if not b'WARNING: Package(s) not found:' in maze:
        envs.extend(["maze-random-5x5-v0", "maze-random-10x10-v0", "maze-random-100x100-v0", 
            "maze-sample-5x5-v0", "maze-sample-10x10-v0", "maze-sample-100x100-v0"])
    else:
        logging.warning("gym-maze not installed, unable to use environments installed with gym-maze package. Install with: pip install -e git+https://github.com/MattChanTK/gym-maze.git#egg=gym-maze")

    if not b'WARNING: Package(s) not found:' in robot_maze:
        envs.extend(["gym_robot_maze:RobotMaze-v1"])
    else:
        logging.warning("gym-robot-maze not installed, unable to use environments installed with gym-robot-maze package. Install with: pip install -e git+https://github.com/finn1y/gym-robot-maze.git#egg=gym-robot-maze")

    return envs

#-----------------------------------------------------------------------------------------------------------
# main
#-----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    #init logging
    logging.basicConfig(format="%(asctime)s.%(msecs)03d: %(levelname)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    if not hasattr(logging, "VDEBUG") and not hasattr(logging, "vdebug") and not hasattr(logging.getLoggerClass(), "vdebug"):
        #add new logging level "vdebug" to logger if not already added
        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.DEBUG - 5):
                self._log(logging.DEBUG - 5, message, args, **kwargs)

        def logToRoot(message, *args, **kwargs):
            logging.log(logging.DEBUG - 5, message, *args, **kwargs)

        logging.addLevelName(logging.DEBUG - 5, "VDEBUG")
    
        setattr(logging, "VDEBUG", logging.DEBUG - 5)
        setattr(logging.getLoggerClass(), "vdebug", logForLevel)
        setattr(logging, "vdebug", logToRoot)

    #list of all possible environements
    envs = ["maze-random-5x5-v0", "maze-random-10x10-v0", "maze-random-100x100-v0", 
            "maze-sample-5x5-v0", "maze-sample-10x10-v0", "maze-sample-100x100-v0", "gym_robot_maze:RobotMaze-v1", 
            "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"]
    #list of all possible algorithms
    #algorithms[:6] are single-agent algorithms (can be used as independent multi-agent algorithms), algorithms[6:] are multi-agent algorithms
    algorithms = ["qlearning", "dqn", "drqn", "policy_gradient", "actor_critic", "ddpg", "ddrqn", "ma_actor_critic"]

    #only provide installed envs on this system as options to user
    args = get_args(get_installed_envs(), algorithms)

    #set more verbose logging level, default is info (verbose == 0)
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.VDEBUG)
    elif args.verbose > 2:
        logging.warning("Maximum verbosity level is 2; logging level set to verbose debug (verbosity level 2).")
        logging.getLogger().setLevel(logging.VDEBUG)

    #check that chosen algorithm can be used with chosen environment action space
    if args.Algorithm == "ddpg" and args.Environment in envs[:10]:
        raise Exception(f'DDPG can only be simulated with continuous action spaces: {envs[9:]}.')
    if args.Algorithm in algorithms[:5] and args.Environment in envs[10:]:
        raise Exception(f'{args.Algorithm} can only be simulated with discrete action spaces: {envs[:9]}.')

    #check that multi-agent algorithms have multiple agents
    if args.Algorithm in algorithms[6:] and args.agents < 2:
        raise Exception(f'{args.Algorithm} is a multi-agent algorithm and must have > 1 agents.')

    #init env
    if args.Environment in envs[:6]:
        env = gym.make(args.Environment, enable_render=args.render, n_robots=args.agents)
    elif args.Environment in envs[6:7]:
        env = gym.make(args.Environment, is_render=args.render, n_agents=args.agents, save_robot_path=True, save_maze_path=args.maze_save_path, load_maze_path=args.maze_load_path)
    else:
        env = gym.make(args.Environment)

    #run algorithm on gym env 
    if args.agents > 1:
        #multi-agent
        if args.Algorithm == "qlearning":
            obvs, actions, rewards, robot_paths = run_gym_q_learning_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "dqn":
            obvs, actions, rewards, losses, robot_paths = run_gym_dqn_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "drqn":
            obvs, actions, rewards, losses, robot_paths = run_gym_dqn_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render, recurrent=True)

        if args.Algorithm == "policy_gradient":
            obvs, actions, rewards, losses, robot_paths = run_gym_policy_grad_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render)
            
        if args.Algorithm == "actor_critic": 
            obvs, actions, rewards, losses, robot_paths = run_gym_actor_critic_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "ddrqn":
            obvs, actions, rewards, losses, robot_paths = run_gym_ddrqn_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "ma_actor_critic":
            obvs, actions, rewards, losses, robot_paths = run_gym_ma_actor_critic_multi_agent(env, n_agents=args.agents, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

    elif args.agents == 1:
        #single-agent
        if args.Algorithm == "qlearning":
            obvs, actions, rewards, robot_paths = run_gym_q_learning_single_agent(env, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "dqn":
            obvs, actions, rewards, losses, robot_paths = run_gym_dqn_single_agent(env, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "drqn":
            obvs, actions, rewards, losses, robot_paths = run_gym_dqn_single_agent(env, episodes=args.episodes, time_steps=args.time_steps, render=args.render, recurrent=True)

        if args.Algorithm == "policy_gradient":
            obvs, actions, rewards, losses, robot_paths = run_gym_policy_grad_single_agent(env, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "actor_critic": 
            obvs, actions, rewards, losses, robot_paths = run_gym_actor_critic_single_agent(env, episodes=args.episodes, time_steps=args.time_steps, render=args.render)

        if args.Algorithm == "ddpg":
            obvs, actions, rewards, losses, robot_paths = run_gym_ddpg_single_agent(env, episodes=args.episodes, time_steps=args.time_steps, render=args.render)
    
    logging.info("Training complete after %u episodes", args.episodes)

    #parameters used (includes defaults even if algorithm does not use that parameter)
    parameters = {"hidden-size": args.hidden_size, "gamma": args.gamma, "epsilon-max": args.epsilon_max, "epsilon-min": args.epsilon_min,
            "lr": args.learning_rate, "decay": args.decay, "batch-size": args.batch_size}

    #q-learning does not have losses
    if "losses" not in locals():
        losses = None

    #process and save captured data
    data = {"Parameters": parameters, "obvs": obvs, "actions": actions, "rewards": rewards, "losses": losses}

    #add agents' path to data if available
    if args.Environment == "gym_robot_maze:RobotMaze-v1":
        data["robot_paths"] = robot_paths
    
    if args.directory:
        save_data(args.directory, data, args.Environment, args.Algorithm)

    if args.plot:
        #plot average reward of agents against episode
        if args.agents > 1:
            avg_reward = [np.average(rewards[i]) for i in range(np.size(rewards))]
            plt.plot(avg_reward)
        else:
            plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Reward")
        plt.show()

    sys.exit(0)

