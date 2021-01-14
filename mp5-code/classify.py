# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because //there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

import numpy as np

# eta is the learning rate


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your c ode here

    train_size = train_set.shape[1]
    W = np.zeros(train_size + 1)
    eta = learning_rate
    y = train_labels

    #print(train_labels)
    
    for cur_iter in range(max_iter):
        for j in range(len(train_set)):
            x = np.concatenate((np.ones(1),train_set[j,:]))
            y_star = np.sign(np.sum(W*x)) # b is added once, it is the first entry
            if(y_star > 0):
                label = 1
            else:
                label = 0
            if(label != y[j]):
                if(y[j] == 1):
                    W += eta*x
                else:
                    W -= eta*x

    return W[1:], W[0]

    # for image in train_set:
    #     # 307 entries per image 
    #     f_sum = 0
    #     j = 0

    #     if train_set == 0:
    #         y_temp = -1
    #     else:
    #         y_temp = 1

    #     y_star = np.sign( np.dot(W, y_temp + b) )
    #     # for pixel in image:
    #     #     f_sum += W[j]*pixel + b
    #     #     j += 1

    #     if(y_star > 0): # IF ANIMAL
    #          label = 1
    #     else:          # IF NOT AN ANIMAL
    #          label = 0

    #     if(label == train_labels[i]):
    #         i += 1
    #         continue
    #     else:
    #         # update weight and bias
    #         j = 0
    #         for pixel in image:
    #             if y[i] == 0:
    #                 W[j] = W[j] + eta*-1*pixel
    #                 if j == 0:
    #                     b = b + eta*-1
    #             else:
    #                 W[j] = W[j] + eta*y[i]*pixel 
    #                 if j == 0:
    #                     b = b + eta*y[i]               
    #             j += 1
            
    #     i += 1

    # return the trained weight and bias parameters 
    # return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    predicts = []

    #raise Exception("BREAK")

    #for cur_iter in range(max_iter):
    for j in range( len(dev_set) ):
        x = dev_set[j,:] # gets all entries at jth block
        f_sum = np.sum(W*x) + b
        if(f_sum > 0):
            predicts.append(1)
        else:
            predicts.append(0)
    return predicts

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    s = 1 / (1 + np.exp(-x))
    
    return s

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters 

    train_size = train_set.shape[1]
    W = np.zeros(train_size + 1)
    eta = learning_rate
    y = train_labels
    print("SHAPE:",train_set.shape[0])

    #print(train_labels)
    
    for cur_iter in range(max_iter):
        gradient_sum = 0 # y_star is gradient
        for j in range(len(train_set)):
            #x = train_set[j,:]
            x = train_set[j]
            x = np.append(x,1)
            
            gradient_sum += ((sigmoid((np.dot(W,x))) - y[j]) * x)
            
        #print("GRADIENT: ",gradient_sum)
        gradient_sum =  gradient_sum 

        W -= eta*gradient_sum / train_set.shape[0]
        
        
    
    return W[:train_size], W[train_size]
    

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    W, b = trainLR(train_set, train_labels, learning_rate, max_iter)
    
    predicts = []
    j = 0
    for x in dev_set:
        #x = dev_set[j,:] # gets all entries at jth block
        f_sum = (sigmoid(np.dot(W,x)+b))
        #print(f_sum) #run time error cause all are zeros
        if(f_sum >= 0.5):
            predicts.append(1)
        else:
            predicts.append(0)
        j +=1
    return predicts
    

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    num_train_set = train_set.shape[0]
    matrix = []
    j = 0
    for dev in dev_set:
        k_nearest_indices = []
    

        matrix.append(np.sum(np.absolute(train_set[i] - dev)))
        j + =1 
        k_nearest = np.argpartition(matrix,k)
        i = 0

        train_labels[k_nearest_indices]
        predicts.append(d)
        

    # W, b = trainLR(train_set,train_labels, 1e-2 ,2)

    # predicts = []
    # j = 0
    # for x in dev_set:
    #     #x = dev_set[j,:] # gets all entries at jth block
    #     f_sum = (sigmoid(np.dot(W,x)+b))
    #     #print(f_sum) #run time error cause all are zeros
    #     if(f_sum >= 0.5):
    #         predicts.append(1)
    #     else:
    #         predicts.append(0)
    #     j +=1
    # return predicts
