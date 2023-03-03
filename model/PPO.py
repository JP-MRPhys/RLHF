import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt


'''
PPOExperience class stores and manages the agent's experiences for the 
Proximal Policy Optimization (PPO) algorithm
'''
class PPOExperience:
    '''
    initializes the PPOExperience object with empty lists for storing experiences.
    '''
    def __init__(self, batch_size):
        self.states = []
        self.log_probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        '''
        generates batches of experiences from the stored lists for training the PPO algorithm
        It first calculates the number of states (n_states) and
        creates an array of indices from 0 to n_states.
        It then shuffles the indices randomly and splits them into batches of size batch_size.
        The method returns the states, actions, log probabilities,
        values, rewards, dones, and batches in numpy arrays.
        '''
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_experience(self, state, action, log_probs, values, reward, done):
        '''
        stores a new experience in the appropriate list
        '''
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_experience(self):
        '''
        clears all stored experiences
        '''
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []


#Train the Agent