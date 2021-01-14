# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

#USE this website as template for both part 1 and 2
#https://pythonprogramming.net/training-deep-learning-neural-network-pytorch/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        ie construct the network architecture
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        The network should have the following architecture (in terms of hidden units):
        in_size -> 128 ->  out_size
        """
        super(NeuralNet, self).__init__() 
        self.loss_fn = loss_fn
        self.fc1 = nn.Linear(in_size, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, out_size)
        self.optimizer = optim.Adam(self.parameters(), lrate)


    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()


    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """ 

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def step(self, x,y):
        """

        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        L = 0.0
        self.zero_grad()
        output = self(x.view(-1,784))
        L = self.loss_fn(output,y)
        L.backward()
        self.optimizer.step() # does the update

        return L.item()

# N is the batch size; D_IN is input dimension
# H is hidden dimension; D_OUT is output dimension, L_RATE is learning rate
#BATCH_SIZE, D_IN, H, D_OUT, L_RATE = 100, 784, 128, 3, 0.001
H = 128 # number of hidden units


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M, in_size) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    criterion = nn.CrossEntropyLoss()
    D_IN = len(train_set[0])
    D_OUT = 3
    BATCH_SIZE, L_RATE = 100,0.001
    BATCHES = int(len(train_set)/BATCH_SIZE)
    net = NeuralNet(L_RATE, criterion, D_IN, D_OUT)
    EPOCHS = int(((n_iter + BATCHES)/ BATCHES))
    print(EPOCHS)
    losses = []
    yhats = []

    for epoch in range(EPOCHS):
        for i in range(int(len(train_set)/BATCH_SIZE)):
            x = train_set[i*batch_size:(i+1)*batch_size]
            y = train_labels[i*batch_size:(i+1)*batch_size]
            
            loss = net.step(x,y)
            losses.append(loss)
 
    print("Finished training")

    with torch.no_grad():
        for data in dev_set:
            output = net(data.view(-1,784))
            for idx, i in enumerate(output):
                yhats.append(torch.argmax(i))

    return losses,yhats, net

    