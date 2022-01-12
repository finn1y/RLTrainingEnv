#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

class ActorCritic():
    def __init__(self, sizes, gamma=0.9, lr=0.0001, net_weights=None):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = sizes[2]

        self.ac_net = ActorCriticNet(sizes)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr) #Adam optimiser is...
        self.loss_fn = tf.keras.losses.Huber() #Huber loss is...

        if net_weights:
            self.ac_net.set_weights(net_weights)

    def get_parameters(self):
        return {"gamma": self.gamma, "lr": self.lr, "net_weights": self.ac_net.get_weights()}

    def get_action(self, obv):
        action_prob = self.ac_net(np.array([obv]))
        action = np.random.choice(self.n_actions, p=action_prob.numpy()[0])

        return int(action)

    def train(self, obv, action, reward, next_obv, done):
        with tf.GradientTape() as tape:
            
            #hmmmmm
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td**2

        grads = tape.gradient(loss, self.ac_net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.ac_net.trainable_variables))

        return loss

class Actor(tf.keras.Model):
    def __init__(self, sizes):
        """
            function to initialise neural network

            sizes is an array of [state_size, hidden_size, action_size] where state_size 
            is the number of inputs, hidden_size is the number of neurons in the hidden 
            layer and action_size is the number of actions
        """
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(sizes[0], activation="relu")
        self.hidden2 = tf.keras.layers.Dense(sizes[1], activation="relu")
        self.actor = tf.keras.layers.Dense(sizes[2], activation="softmax")
        self.critic = tf.keras.layers.Dense(1, activation="linear")

    def call(self, obv):
        """
            function to define the forward pass of the neural network this function is called 
            when ActorCriticNet(inputs) is called or ActorCriticNet.predict(inputs) is called

            obv is the numpy array or tensor of the inputs values to the neural network
        """
        obv = self.hidden1(obv)
        obv = self.hidden2(obv)

        policy = self.actor(obv)
        value = self.critic(obv)

        return policy, value

