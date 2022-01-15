#!/usr/bin/env python3

import numpy as np

class QLearning():
    def __init__(self, sizes, gamma=0.9, epsilon_max=0.8, epsilon_min=0.1, lr=0.7, lr_decay=0.95):
        """
            function to initalise the QLearning class

            sizes is an array of [observations, actions] where observations is an array of 
            [observations_low, observations_high] and actions is an array of [actions_low, actions_high]
            in turn low is an array of low bounds for each observation/action and high is an array of high
            bounds for each observation/action respectively

            gamma is a float which is the discount factor of future rewards

            epsilon max is a float which is the maximum exploration rate of the agent

            epsilon min is a float which is the minimum exploration rate of the agent

            epsilon decay is a float which is the rate at which epsilon will decrease

            lr is a float which is the learning rate of the agent

            lr_decay is a float which is the rate at which the learning rate will decay exponentially
        """
        self.gamma = gamma
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon_max = epsilon_max 
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.high = np.array(sizes[0][1], dtype=int)
        self.n_actions = np.shape(sizes[1])[1]

        self.q_table = np.zeros((np.sum(self.high + 1), self.n_actions))
        print(self.q_table)

    def get_parameters(self):
        """
            function to get the parameters of the algorithm

            returns a dict with all the algorithm parameters
        """
        return {"gamma": self.gamma, "epsilon_max": self.epsilon_max, "epsilon_min": self.epsilon_min, "lr": self.lr, "lr_decay": self.lr_decay}

    def get_action(self, obv):
        """
            function to get the action based on the current observation using an epsilon-greedy policy

            obv is the current observation of the state

            update epsilon is a bool to determine if to update epsilon's value

            returns the action to take as an int
        """
        if np.size(self.high) > 1:
            obv = self.__index_obv__(np.array(obv, dtype=int))
        else:
            obv = int(obv)

        #take random action with probability epsilon (explore rate)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)

        else:
            #policy is greedy
            action = np.argmax(self.q_table[obv])

        return action

    def update_parameters(self, n_t, n_max):
        """
            function to reduce value of learning parameters such that epsilon is epsilon max at n_t = 0 and 
            epsilon min at n_t = n_max, and learning rate decays exponentially with a rate of learning rate decay

            n_t is the current episode number

            n_max is the maximum number of epsiodes
        """
        rate = max((n_max - n_t) / n_max, 0) #rate should not be less than zero
        self.epsilon = rate * (self.epsilon_max - self.epsilon_min) + self.epsilon_min

        self.lr *= self.lr_decay ** n_t

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

