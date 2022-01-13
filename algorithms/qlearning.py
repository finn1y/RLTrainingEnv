#!/usr/bin/env python3

import numpy as np

class QLearning():
    def __init__(self, sizes, gamma=0.9, epsilon=1.0, epsilon_decay=0.0000005, lr=0.0001):
        """
            function to initalise the QLearning class

            sizes is an array of [observations, actions] where observations is an array of 
            [observations_low, observations_high] and actions is an array of [actions_low, actions_high]
            in turn low is an array of low bounds for each observation/action and high is an array of high
            bounds for each observation/action respectively

            gamma is a float which is the discount factor of future rewards

            epsilon is a float which is the exploration rate of the agent

            epsilon decay is a float which is the rate at which epsilon will decrease exponentially

            lr is a float which is the learning rate of the agent
        """
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.high = sizes[0][1]
        self.n_actions = np.shape(sizes[1])[1]

        self.q_table = np.zeros((np.sum(self.high + 1), self.n_actions))

    def get_parameters(self):
        """
            function to get the parameters of the algorithm

            returns a dict with all the algorithm parameters
        """
        return {"gamma": self.gamma, "epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay, "lr": self.lr}

    def get_action(self, obv):
        """
            function to get the action based on the current observation using an epsilon-greedy policy

            obv is the current observation of the state

            update epsilon is a bool to determine if to update epsilon's value

            returns the action to take as an int
        """
        if np.size(self.high) > 1:
            obv = self.__index_obv__(obv)

        #take random action with probability epsilon (explore rate)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)

        else:
            #policy is greedy
            action = np.argmax(self.q_table[obv])

        return action

    def update_epsilon(self):
        """
            function to update epsilon using exponential decay
        """
        self.epsilon += 1 - np.exp(self.epsilon_decay)
        self.epsilon = 0.1 if self.epsilon < 0.1 else self.epsilon #minimum value for epsilon is 0.1

    def train(self, obv, action, reward, next_obv):
        """
            function to train agent 
        """
        if np.size(self.high) > 1:
            obv = self.__index_obv__(obv)
            next_obv = self.__index_obv__(next_obv)

        self.q_table[obv, action] += self.lr * (reward + self.gamma * np.max(self.q_table[next_obv]) - self.q_table[obv, action])

    def __index_obv__(self, obv) -> int:
        index_obv = 0

        for i in range(np.size(self.high)):
            offset = 0 if i == 0 else int(np.sum(self.high[:i - 1]))
            index_obv += int(obv[i]) + offset

        return index_obv


