#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

class PolicyGradient():
    """
        Class to contain the PolicyNetwork and all parameters
    """
    def __init__(self, sizes, gamma=0.9, lr=0.0001, saved_path=None):
        """
            function to initialise the class

            sizes is an array of [number_of_inputs, hidden_layer_neurons, number_of_outputs] 
            where number of inputs is equivalent to size of a state, hidden layer neurons is 
            the number of neurons in the hidden layer and number of outputs is equivalent to
            the number of possible actions

            gamma is the discount factor of future rewards

            lr is the learning rate of the neural network

            saved_path is a string of the path to the saved Actor-Critic network if one is being loaded
        """
        self.gamma = gamma
        self.lr = lr
        self.n_actions = sizes[2]
        self.replay_mem = []
        self.eps = np.finfo(np.float32).eps.item()

        self.policy_net = PolicyNet(sizes)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr) #Adam optimiser is...

        #load a saved model (neural net) if provided
        if saved_path:
            self.policy_net = tf.keras.models.load_model(saved_path, custom_object={"CustomModel": PolicyNet})

    def get_parameters(self):
        """
            function to get the parameters of the algorithm

            returns a dict with all the algorithm parameters
        """
        return {"gamma": self.gamma, "lr": self.lr}

    def save_model(self, path):
        """
            function to save the tensorflow model (neural net) to a file

            path is a string of the path to the file where the model will be saved
        """
        self.policy_net.save(path)

    def get_action(self, obv):
        """
            function to get the action based on the current observation using the 
            policy generated by the neural net

            obv is the current observation of the state

            returns the action to take
        """
        action_probs = self.policy_net(np.array([obv]))
        action = np.random.choice(self.n_actions, p=action_probs.numpy()[0])
        
        return action

    def store_episode(self, obv, action, reward, next_obv):
        """
            function to store an episode's tuple of values

            obv is the observation of the current state

            action is an int of the action taken

            reward is the reward returned when the action is applied to the current state

            next obv is the observation of the next state after action has been applied to the current state
            (placeholder for compatibility across algorithm classes)
        """
        #next_obv is not used in model training 
        self.replay_mem.append({"obv": obv, "action": action, "reward": reward})

    def train(self):
        """
            function to train Policy network using previous episode data from replay memory

            returns the loss of the training as a tensor
        """
        obv_batch = np.array([self.replay_mem[i]["obv"] for i in range(np.shape(self.replay_mem)[0])])
        action_batch = np.array([self.replay_mem[i]["action"] for i in range(np.shape(self.replay_mem)[0])])
        returns = []
        discounted_sum = 0

        #calculate the discounted sum of rewards
        for step in self.replay_mem[::-1]:
            discounted_sum = step["reward"] + self.gamma * discounted_sum
            #iterated inversly therefore insert at beginning of array
            returns.insert(0, discounted_sum)

        #normalise returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)

        with tf.GradientTape() as tape:
            action_probs = self.policy_net(obv_batch)

            loss = 0
            for i in range(np.shape(action_probs)[0]):
                #log probability of the action taken
                action_log_prob = tf.math.log(action_probs[i, action_batch[i]])
                #sum loss across episode
                loss += -action_log_prob * returns[i]

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        #replay memory only stores a single episode 
        self.replay_mem.clear()

        return loss

class PolicyNet(tf.keras.Model):
    """
        Class to contain the neural network approximating the policy
    """
    def __init__(self, sizes):
        """
            function to initialise the class

            sizes is an array of [number_of_inputs, hidden_layer_neurons, number_of_outputs] 
            where number of inputs is equivalent to size of a state, hidden layer neurons is 
            the number of neurons in the hidden layer and number of outputs is equivalent to
            the number of possible actions
        """
        super(PolicyNet, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(sizes[0], activation="relu")
        self.hidden2 = tf.keras.layers.Dense(sizes[1], activation="relu")
        self.policy = tf.keras.layers.Dense(sizes[2], activation="softmax")

    def call(self, obv):
        """
            function to define the forward pass of the neural network this function is called 
            when QNet(inputs) is called or QNet.predict(inputs) is called

            obv is the numpy array or tensor of the inputs values to the neural network

            returns a tensor of the probability distribution of the policy output by the neural network
        """
        obv = self.hidden1(obv)
        obv = self.hidden2(obv)
        policy = self.policy(obv)

        return policy

