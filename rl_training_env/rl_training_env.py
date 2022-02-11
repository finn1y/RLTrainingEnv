#!/usr/bin/env python

import os, sys, subprocess
import argparse, pickle
import gym
import gym_maze
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from algorithms import *

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
    parser.add_argument("-a", "--agents", type=int, default=1, help="Number of agents")
    parser.add_argument("-d", "--directory", type=str, default=None, help="Save the results from the training to the specified directory")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor to multiply by future expected rewards in the algorithm. Should be greater than 0 and less than 1")
    parser.add_argument("--epsilon-max", type=float, default=1.0, help="Epsilon max is the intial value of epsilon for epsilon-greedy policy. Should be greater than 0 and less than or equal to 1")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Epsilon min is the final value of epsilon for epsilon-greedy policy which is decayed to over training from epsilon max. Should be greater than 0 and less then epsilon max")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.0001, help="Learning rate of the algorithm. Should be greater than 0 and less than 1")
    parser.add_argument("--learning-rate-decay", type=float, default=0.95, help="Learning rate decay the base used for exponential decay of the learning rate during training if set to 1 will have no decay. Should be greater than 0 and less than 1")

    return parser.parse_args()

def save_data(path, data, agent, env, algorithm):
    """
        function to save data in a pickle file gathered during training in the directory at path
        directory has the following structure, for q-learning:

            -path
                -data.pkl
                -q_table.pkl

        or for any other algorithm:

            -path
                -data.pkl
                -agent0
                    -(saved tf model)
                -agent1
                    -(saved tf model)

        path is a string of the path to the directory to store the data in

        data is a dictionary of the data to be saved

        agent is a reference of the agent trained on the environment with the algorithm

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

    #model (or q-table) saved for continuation of training if desired
    if algorithm != "qlearning":
        #save tf model
        for i in range(np.size(agents)):
            agents[i].save_model(f'{path}/agent{i}')
    else:
        #save q-table in pickle file
        for i in range(np.size(agents)):
            with open(f'{path}/q_table{i}.pkl', "wb") as handle:
                pickle.dump(agents[i].q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_installed_envs():
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
        print("WARNING: gym not installed, unable to use environments installed with gym package. Install with: pip install gym")

    if not b'WARNING: Package(s) not found:' in maze:
        envs.extend(["maze-random-5x5-v0", "maze-random-10x10-v0", "maze-random-100x100-v0", 
            "maze-sample-5x5-v0", "maze-sample-10x10-v0", "maze-sample-100x100-v0"])
    else:
        print("WARNING: gym-maze not installed, unable to use environments installed with gym-maze package. Install with: pip install -e git+https://github.com/MattChanTK/gym-maze.git#egg=gym-maze")

    if not b'WARNING: Package(s) not found:' in robot_maze:
        envs.extend(["gym_robot_maze:robot-maze-v0"])
    else:
        print("WARNING: gym-robot-maze not installed, unable to use environments installed with gym-robot-maze package. Install with: pip install -e git+https://github.com/finn1y/gym-robot-maze.git#egg=gym-robot-maze")

    return envs

if __name__ == "__main__":
    #list of all possible environements
    envs = ["maze-random-5x5-v0", "maze-random-10x10-v0", "maze-random-100x100-v0", 
            "maze-sample-5x5-v0", "maze-sample-10x10-v0", "maze-sample-100x100-v0", "gym_robot_maze:robot-maze-v0", 
            "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"]
    #list of all possible algorithms
    #algorithms[:6] are single-agent algorithms (can be used as independent multi-agent algorithms), algorithms[6:] are multi-agent algorithms
    algorithms = ["qlearning", "dqn", "drqn", "policy_gradient", "actor_critic", "ddpg", "ddrqn", "ma_actor_critic"]

    #only provide installed envs on this system as options to user
    args = get_args(get_installed_envs(), algorithms)

    #check that chosen algorithm can be used with chosen environment action space
    if args.Algorithm == "ddpg" and args.Environment in envs[:10]:
        raise Exception(f'DDPG can only be simulated with continuous action spaces: {envs[9:]}')
    if args.Algorithm in algorithms[:5] and args.Environment in envs[10:]:
        raise Exception(f'{args.Algorithm} can only be simulated with discrete action spaces: {envs[:9]}')

    #check that multi-agent algorithms have multiple agents
    if args.Algorithm in algorithms[6:] and args.agents < 2:
        raise Exception(f'{args.Algorithm} is a multi-agent algorithm and must have >1 agents')

    if args.Environment in envs[:6]:
        env = gym.make(args.Environment, enable_render=args.render, n_robots=args.agents)
    elif args.Environment in envs[6:7]:
        env = gym.make(args.Environment, is_render=args.render, n_agents=args.agents, save_robot_path=True)
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
    
    #create list of agent objects depending on the algorithm to be used
    if args.Algorithm == "qlearning":
        agents = [QLearning([observations, actions], gamma=args.gamma, epsilon_max=args.epsilon_max, epsilon_min=args.epsilon_min, lr=args.learning_rate, lr_decay=args.learning_rate_decay) for i in range(args.agents)]

    if args.Algorithm in algorithms[1:3]:
        #drqn is a deep q-network with a long short term memory replacing the first layer of the q-network 
        recurrent = True if args.Algorithm == "drqn" else False
        agents = [DQN([observations, args.hidden_size, actions], gamma=args.gamma, epsilon_max=args.epsilon_max, epsilon_min=args.epsilon_min, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps,  DRQN=recurrent, saved_path=args.model_path) for i in range(args.agents)]

    if args.Algorithm == "policy_gradient":
        agents = [PolicyGradient([observations, args.hidden_size, actions], gamma=args.gamma, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps, saved_path=args.model_path) for i in range(args.agents)]

    if args.Algorithm == "actor_critic": 
        agents = [ActorCritic([observations, args.hidden_size, actions], gamma=args.gamma, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps, saved_path=args.model_path) for i in range(args.agents)]

    if args.Algorithm == "ddpg":
        agents = [DDPG([observations, args.hidden_size, actions], gamma=args.gamma, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps, saved_path=args.model_path) for i in range(args.agents)]
    
    if args.Algorithm == "ddrqn":
        agents = [DDRQN([observations, args.hidden_size, actions], gamma=args.gamma, epsilon_max=args.epsilon_max, epsilon_min=args.epsilon_min, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps, saved_path=args.model_path) for i in range(args.agents)]

    if args.Algorithm == "ma_actor_critic":
        agents = [MAActorCritic([observations, args.hidden_size, actions], gamma=args.gamma, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps, n_agents=args.agents, saved_path=args.model_path) for i in range(args.agents - 1)]
        #agent 0 is the master the rest are slaves
        agents.insert(0, MAActorCritic([observations, args.hidden_size, actions], gamma=args.gamma, lr=args.learning_rate, lr_decay=args.learning_rate_decay, lr_decay_steps=args.time_steps, n_agents=args.agents, master=True, saved_path=args.model_path))

    all_losses = []
    all_rewards = []

    if args.Environment in envs[6:7]:
        #robot-maze env can save the path taken by the agents each episode
        robot_paths = []

    if args.render:
        env.render()
    
    for e in range(args.episodes):
        obvs = env.reset()
        
        if args.Environment in envs[7:]:
            #environment not multi-agent compatible 
            obvs = np.array([obvs]) #wrap obvs in array

        ep_losses = []
        total_rewards = np.zeros(args.agents)
        done = False

        for t in range(args.time_steps):
            if args.render:
                env.render()

            if args.Algorithm == "ddpg":
                #continuous action space requires float type actions
                actions = np.zeros(args.agents, dtype=np.float32)
            else:
                #dicrete action space requires int type actions
                actions = np.zeros(args.agents, dtype=int)

            for i in range(args.agents):
                actions[i] = agents[i].get_action(obvs[i])
            
            if args.Environment in envs[:7] or args.Environment in envs[10:12]:
                #continuous action envs require action in an array
                next_obvs, rewards, done, info = env.step(actions)

                if args.Environment in envs[10:12]:
                    #for continuous action space, single action envs: wrap next_obvs and rewards in arrays
                    next_obvs = np.array([next_obvs])
                    rewards = np.array([rewards])

            else:
                #environment not multi-agent compatible 
                next_obvs, rewards, done, _ = env.step(actions[0])
                #wrap next_obvs and rewards in arrays
                next_obvs = np.array([next_obvs])
                rewards = np.array([rewards])

            for i in range(args.agents):
                if args.Algorithm == "qlearning" or args.Algorithm == "ddrqn":
                    #q-learning and ddrqn do not use experience relay of any kind
                    agents[i].train(obvs[i], actions[i], rewards[i], next_obvs[i])

                    if args.Algorithm == "ddrqn":
                        #each agent sends their updated weights to the next agent for the next update
                        j = (i + 1) % (args.agents - 1)
                        agents[j].receive_comm(agents[i].send_comm())

                if args.Algorithm in algorithms[1:6] or args.Algorithm in algorithms[7:8]:
                    #all other algorithms use some form of experience replay
                    agents[i].store_step(obvs[i], actions[i], rewards[i], next_obvs[i])

            if args.Algorithm in algorithms[6:]:
                for i in range(1, args.agents):
                    if args.Algorithm == "ma_actor_critic":
                        #master agent recieves each other agents gathered information for that time step
                        agents[0].receive_comm(agents[i].send_comm())

                    if args.Algorithm == "ddrqn":
                        #agent 0 has the most up to date network and should update all other agents networks
                        agents[i].receive_comm(agents[0].send_comm())

            obvs = next_obvs
            total_rewards += rewards

            if (args.Algorithm in algorithms[1:3] or args.Algorithm == "ddpg") and t >= batch_size and t % 4 == 0:
                #for expereince replay using a batch size train agents once batch size is reached and then every 4 timesteps
                losses = []

                for i in range(args.agents):    
                    loss = agents[i].train(batch_size)
                    losses.append(loss)    

                    if t % 20 == 0:
                        #update the target network every 20 timesteps after the batch size is reached
                        agents[i].update_target_net()

                ep_losses.append(loss)

            if done or t == (args.time_steps - 1):
                #final timestep either goal state is reached or max timesteps reached
                print(f'Episode {e} finished after {t} time steps with total rewards = {total_rewards},')

                if args.Algorithm == "ddrqn":
                    #ddrqn updates target network at the end of the episode
                    for i in range(args.agents):
                        agents[i].update_target_net()

                if args.Algorithm in algorithms[:3] or args.Algorithm == "ddrqn":
                    #algorithms using epsilon greedy policy must update value of epsilon
                    for i in range(args.agents):
                        agents[i].update_parameters(e, args.episodes)
                
                if args.Algorithm in algorithms[3:5]:
                    #for experience replay using previous episode train agents at end of episode
                    losses = []

                    for i in range(args.agents):
                        loss = agents[i].train()
                        losses.append(loss)

                    ep_losses.append(losses)

                if args.Algorithm in algorithms[7:8]:
                    #for experience replay using previous episode train agents at end of episode
                    losses = []

                    #train actor and critic network in master
                    loss = agents[0].train()
                    losses.append(loss)

                    for i in range(1, args.agents):
                        #slaves train actor network using value from master critic network
                        agents[i].receive_comm(agents[0].send_comm())

                        loss = agents[i].train()
                        losses.append(loss)

                    ep_losses.append(loss)

                if args.Environment == "gym_robot_maze:robot-maze-v0":
                    robot_paths.append(info["robot_path"])

                all_rewards.append(total_rewards)
                all_losses.append(ep_losses)
                break

            if args.Environment in envs[:6] and env.is_game_over():
                sys.exit(0)

    print(f'Training complete after {args.episodes} episodes')

    #parameters are the same for all agents so only need to save a single agents parameters
    data = {"Parameters": agents[0].get_parameters(), "rewards": all_rewards, "losses": all_losses}

    #add agents' path to data if available
    if args.Environment == "gym_robot_maze:robot-maze-v0":
        data["robot_paths"] = robot_paths
    
    if args.directory:
        save_data(args.directory, data, agents, args.Environment, args.Algorithm)

    if args.plot:
        #plot average reward of agents against episode
        avg_reward = [np.average(data["rewards"][i]) for i in range(np.shape(data["rewards"])[0])]
        plt.plot(avg_reward)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Reward")
        plt.show()

    sys.exit(0)

