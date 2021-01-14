# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter
#import pickle
#pickle.dump(open( ))

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """


    # TODO: Write your code here

    def laplace_smooth(count, k, N, k_x):
        return math.log( (count + k) / (N + k_x) )

    # if pos_prior == 1:
    #     return [1 for i in range(len(dev_set))]
    # elif pos_prior == 0:
    #     return [0 for i in range(len(dev_set))]
    
    #word_map = {}

    list_dev = []

    pos_counter = Counter()
    neg_counter = Counter()
    pos_word_count = 0
    neg_word_count = 0
    # go through all reviews in training set
    for i in range(len(train_set)):
        cur_list = train_set[i].copy()
        #word_inner_list.clear()
        # go through each word in the review
        for j in range(len(cur_list) - 1):
            if(train_labels[i]): # check if positive review
                pos_counter.update(cur_list[j]) 
                pos_word_count += 1
            else:
                neg_counter.update(cur_list[j])
                neg_word_count += 1

    #counter_dev = Counter()


    for cur_review in dev_set:
        rev_count = Counter(cur_review)
        prob_pos = math.log(pos_prior)
        prob_neg = math.log(1 - pos_prior)
        #rev_total_words = len( list(rev_count.keys() ))

        for cur_key in rev_count:
            #prob_word_pos = math.log( pos_counter.get(cur_key,0) / pos_word_count )
            prob_laplace_smooth_pos = ( laplace_smooth(pos_counter.get(cur_key,0)
                ,smoothing_parameter, pos_word_count, len(list(pos_counter.keys()))))

            #prob_word_neg = math.log( neg_counter.get(cur_key,0) / neg_word_count )
            prob_laplace_smooth_neg = ( laplace_smooth(neg_counter.get(cur_key,0)
                ,smoothing_parameter, neg_word_count, len(list(neg_counter.keys())))) 
                
            prob_pos += prob_laplace_smooth_pos
            prob_neg += prob_laplace_smooth_neg

        if(prob_pos > prob_neg):
            list_dev.append(1)
        else:
            list_dev.append(0)



    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return list_dev