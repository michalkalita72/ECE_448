import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy




class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Stores the Q-value for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
    
        self.buckets = buckets
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        #WHERE Q-VAL TASBLE IS STORED
        #self.model = np.zeros(self.buckets + (actionsize,)) # numpy array
        if(model is None):
            self.model = np.zeros(self.buckets + (actionsize,))
            self.table = np.zeros(self.buckets + (actionsize,))
        else:
            self.model = model
            self.table = np.zeros(self.buckets + (actionsize,))
        self.env = env

    def discretize(self, obs):
        """
        Discretizes the continuous input observation, observation - state(position, velocity)

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the list of q values for the state for each action. 
        """
        #print("DISCRETE STATE AND TYPE:", states, list(states[0]), type(list(states[0])))
        discrete_state = self.discretize(list(states[0]))
        qval_l = self.model[discrete_state]
        #raise Exception("Q-VAL:", qval_l)
        #qval_l = np.array(qval_l).flatten().tolist()
        #raise Exception(type(qval_l))
        return qval_l, discrete_state

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
       
        C = 100
        discrete_state = self.discretize(state)
        discrete_next_state = self.discretize(next_state)
        current_qval = self.model[discrete_state][action]
       
        future_qval = np.max(self.model[discrete_next_state])
       
        if(done == True and next_state[0] >= 0.5):
            target = 1.0 
        else:
            target = reward + self.gamma * future_qval

        self.table[discrete_state][action] += 1

        lr = min(self.lr, C / (C+ self.table[discrete_state][action]))
        new_q = current_qval + lr * (target - current_qval)


        self.model[discrete_state][action] = new_q

        return (target - new_q)**2

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('MountainCar-v0')

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(30, 20), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/tabular.npy')
