import numpy as np
import math
from collections import Counter


"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    def laplace_smooth(count, k, N, k_x):
        return math.log( (count + k) / (N + k_x) )
    # tagset = {'NOUN', 'VERB', 'ADJ', 'ADV',
    #       'PRON', 'DET', 'IN', 'NUM',
    #       'PART', 'UH', 'X', 'MODAL',
    #       'CONJ', 'PERIOD', 'PUNCT', 'TO', 'UNKOWN'}
    alpha = 0.000001
    predicts = []  # The test list but with tags attached to the words
    #raise Exception("You must implement me")

    wordCount = Counter()
    tag_count = Counter()

    for sentence in train: 
        arr = np.array(sentence)
        tag_count.update(arr[:,1])
        wordCount.update(sentence)
    
        

    #max_tag_count = 0
    j = 0 # count for how many times a tag wasn't attached to a word
    topTag = (tag_count.most_common(1))[0][0] # most common tag in entire data set
    # topTag raw form: [(('the', 'DET'), 43627)]
    tagset = list(tag_count.keys())
    #print(topTag)
    for sentence in test:
        
        temp_list = []
        for word in sentence:
                max_tag_count = 0
                j = 0
                tagNum = [wordCount.get((word,tag),0.) for tag in tagset]
                if( max(tagNum) == 0.):
                    best_tag = topTag
                else:
                    best_tag =  tagset[np.argmax(tagNum)]
                # for tag in tagset:
                #     if( wordCount[ (word,tag) ] > max_tag_count):
                #         max_tag_count = wordCount[ (word,tag) ]
                #         best_tag = tag
                # if(max_tag_count == 0):
                #     best_tag = topTag
                
                #print(wordCount[ tuple(pair) ] )
                temp_list.append( (word,best_tag))
        predicts.append(temp_list)
                                
    #print(len(predicts))
    return predicts

# VITERBI PT 1
def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    def laplace_smooth(count, k, N, k_x):
        return math.log( (count + k) / (N + k_x) )
    k = 0.00001

    predicts = []
    #raise Exception("You must implement me")

    wordTag_count = Counter() # Counts num of times a given word tag pair appears in training set
    Tag_count = Counter() # Counts number of times a given tag appears in training set
    Tag_pair_count = Counter() # Counts number of times a tag_cur follows tag_prev in training set
    wordTag_initCount = Counter() # of times a tag is at the start of a sentence

    vocab= set() # Counts the number of distinct words 

    num_Tags = 0 # total number of tags that appear in training set
    num_startin_pos = len(train) # total number of starting positions

    # COUNTING
    for sentence in train:
        vocab.update(sentence)
        i = 0
        wordTag_initCount[sentence[0]] += 1
        for pair in sentence:
           #print(pair)
            wordTag_count[ pair ] += 1
            Tag_count[ pair[1] ] += 1
            num_Tags += 1
            if(i != 0):
                prev_tag = cur_tag 
                cur_tag = pair[1]
                Tag_pair_count[ (prev_tag, cur_tag)] += 1
                #print("NUM TIMES TAG PAIR: ", prev_tag, " ", cur_tag, " ", Tag_pair_count[(prev_tag,cur_tag)])
            else:
                i += 1
                cur_tag = pair[1]
                continue # no previous tag at first word
            i += 1
            

    vocab_size = len(vocab) # total number of distinct words

    #print("NUM WORD/TAG PAIR: ", wordTag_count[ ('the','DET') ])
    #print("NUM TAGS: ", vocab_size)
    #print("NUM TIMES TAG: ", pair[1], " ", Tag_count[pair[1]] )
    #print("NUM TIMES TAG PAIR:", "NOUN->NOUN", Tag_pair_count[("NOUN", "NOUN")])
   
    
    # CALCULATING THE PROBABILITIES

    init_probability_dict = {} # stores the initial probability for each word/Tag pair
    emit_probability_dict = {} # stores the emission probability of each word/Tag pair
    trans_probability_dict = {} # stores the transition probability for each tag pair
    parent_dict = {} # stores the word/tag pair for an index

    i = 0 # represents index of sentence
    for sentence in test:
        i = 0
        for word in sentence:

                # INITIAL PROBABILITIES
            for tag in Tag_count.keys():
                if(i == 0):
                    k_n = k * num_Tags 
                    pair = (word,tag)
                    init_probability_dict[pair]  = laplace_smooth( wordTag_initCount[pair], k, num_startin_pos, k_n)
                # EMISSION PROBABILITIES
                else:
                    k_n = k * (vocab_size + 1)
                    pair = (word,tag)
                    emit_probability_dict[pair] = laplace_smooth(wordTag_count[pair], k, Tag_count[tag], k_n)
            i +=1

    # TRANSITION PROBABILITIES
    for tag_prev in Tag_count.keys():
        for tag_cur in Tag_count.keys():
            k_n = k * num_Tags
            tag_pair = (tag_prev,tag_cur)
            trans_probability_dict[tag_pair] = laplace_smooth(Tag_pair_count[tag_pair],k,Tag_count[tag_prev],k_n)


    # test_pair = ('DET','NOUN')
    # print("PROB DET-> NOUN: ", trans_probability_dict[test_pair])
    # print("Count(tag_prev->tag_cur): ",Tag_pair_count[test_pair])
    # print("COUNT(tag_prev): ", Tag_count['DET'])
    # print("NUM_OF_TAGS: ", num_Tags)
    # print("K: ", k)
    # print(math.log(2))

    # test_pair = ('said','VERB')
    # print("PROB count, NOUN: ", init_probability_dict[test_pair])
    # print("Count(tag, start): ",Tag_pair_count[test_pair])
    # print("COUNT(starting pos): ", num_startin_pos)
    # print(k*num_Tags)

    # TRELLIS
    
    # n = 3
    # m = len(Tag_count.keys())
    # trellis = [ [[smallest_num,None] for v in range(m)] for u in range(n) ]
    # pair = ('the','DET')
    # trellis[0][0] = [init_probability_dict.get(pair) * emit_probability_dict.get(pair), None]
    # print(trellis[0][0])
    # raise Exception("BREAK")
    i = 0 # current index of sentence
    j = 0 # current index of Tag_count.keys()
    for sentence in test:
        i = 0
        n = len(sentence)
        m = len(Tag_count.keys())
        trellis = [ [[0,None] for v in range(m)] for u in range(n) ]
        sub_predicts = []
        for word in sentence:
            j = 0
            for tag in Tag_count.keys():
                # STARTING POSITION OF SENTENCE
                pair = tuple((word,tag))
                if(i == 0): # NO parent = None
                    #print(word, " ",tag)
                    #print(init_probability_dict.get(pair, 0) + emit_probability_dict.get(pair,0), " ", word, " ", tag)
                    block = init_probability_dict.get(pair,0) + emit_probability_dict.get(pair,0)
                    trellis[i][j] = [block, None]
                    parent_dict[(i,j)] = (word,tag)
                    #print(trellis[i][j])
                # AFTER STARTING POSITION
                else:
                    transition_list =[]
                    z = 0 # another counter for tag index
                    #print(i-1)
                    #print(trellis[i-1])
                    parent_dict[(i,j)] = (word,tag)
                    for node_value,prev_tag in zip(trellis[i - 1], Tag_count.keys()): # find max transition
                        tag_tup = (prev_tag,tag)
                        transition_list.append(node_value[z] + trans_probability_dict[(tag_tup)])
                    max_trans_prob = max(transition_list)
                    max_trans_idx = np.argmax(transition_list)
                    # print(transition_list)
                    # print(max_trans_idx)
                    # print(max_trans_prob)
                    trellis[i][j] = [emit_probability_dict[pair]+max_trans_prob, (i -1,max_trans_idx)]
                j += 1
            i += 1

        # END OF SENTENCE, END OF INNER FOR LOOP
        # FIND THE MAX TERMINAL NODE AND ASSIGN APPROPRIATE WORD/TAG PAIR
        terminal_node_list = []
        terminal_nodeIdx_list = []
        for node_terminal, tag_2 in zip(trellis[len(sentence)-1],Tag_count.keys()):
            terminal_node_list.append(node_terminal[0])
            terminal_nodeIdx_list.append(node_terminal[1])
        max_terminal_node = max(terminal_node_list) # Get max terminal node probability
        max_terminal_node_idx = np.argmax(terminal_node_list) # index of max node in terminal_node_list
        parent = terminal_nodeIdx_list[max_terminal_node_idx] # parent of max terminal node
        sub_predicts.append( parent_dict[( len(sentence) -1, max_terminal_node_idx)] )
        #print("TRELLIS: ", trellis[len(sentence)-1])
        #print("MAX_NODE ", max_terminal_node, "IDX", max_terminal_node_idx, "PAR",parent)
        
        # TRACE BACKWARDS FROM  TERMINAL NODE
        while(parent!= None):
            temp_i = parent[0]
            temp_j = parent[1]
            parent_node = trellis[temp_i][temp_j]
            word_tag = parent_dict[ (temp_i,temp_j) ]
            sub_predicts.append(word_tag)
            parent = parent_node[1] #get parent of next node
        sub_predicts.reverse()
        #print(sub_predicts)
        predicts.append(sub_predicts)

    #raise Exception("BREAK")
    return predicts

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    def laplace_smooth(count, k, N, k_x):
        return math.log( (count + k) / (N + k_x) )
    k = 0.00001

    predicts = []
    #raise Exception("You must implement me")

    wordTag_count = Counter() # Counts num of times a given word tag pair appears in training set
    Tag_count = Counter() # Counts number of times a given tag appears in training set
    Tag_pair_count = Counter() # Counts number of times a tag_cur follows tag_prev in training set
    wordTag_initCount = Counter() # of times a tag is at the start of a sentence
    word_count = Counter() # counts number of times a word appeared
    wordTag_once_count = Counter() # For a given tag, count number of times a word appears once for it
    word_dict = {}

    vocab= set() # Counts the number of distinct words 
    vocab_tag = set()

    num_Tags = 0 # total number of tags that appear in training set
    num_startin_pos = len(train) # total number of starting positions
    words_occurring_once = 0 # total number of words in train occurring once

    # COUNTING
    for sentence in train:
        vocab.update(sentence)
        i = 0
        wordTag_initCount[sentence[0]] += 1
        for pair in sentence:
           #print(pair)
            word_dict[pair[0]] = pair[0]
            wordTag_count[ pair ] += 1
            Tag_count[ pair[1] ] += 1
            word_count[pair[0]] += 1
            num_Tags += 1
            if(i != 0):
                prev_tag = cur_tag 
                cur_tag = pair[1]
                Tag_pair_count[ (prev_tag, cur_tag)] += 1
                #print("NUM TIMES TAG PAIR: ", prev_tag, " ", cur_tag, " ", Tag_pair_count[(prev_tag,cur_tag)])
            else:
                i += 1
                cur_tag = pair[1]
                continue # no previous tag at first word
            i += 1
            
    for word,pair in zip(word_count,wordTag_count):
        if(wordTag_count[pair] == 1):
            wordTag_once_count[pair[1]] += 1
            #print("COUNT(words_occurring_once,tag)",wordTag_once_count.values())
        if(word_count[word] == 1):
            words_occurring_once +=1 
    
    
    #print("WORDS OCCURRING ONCE:", words_occurring_once)
   
    vocab_size = len(vocab) # total number of distinct words
   
    # CALCULATING THE PROBABILITIES

    init_probability_dict = {} # stores the initial probability for each word/Tag pair
    emit_probability_dict = {} # stores the emission probability of each word/Tag pair
    trans_probability_dict = {} # stores the transition probability for each tag pair
    parent_dict = {} # stores the word/tag pair for an index
    hapax_probability_dict = {}

    i = 0 # represents index of sentence
    for sentence in test:
        i = 0
        for word in sentence:

                # INITIAL PROBABILITIES
            for tag in Tag_count.keys():
                pair = (word,tag)
                if(i == 0):
                    k_n = k * num_Tags 
                    init_probability_dict[pair]  = laplace_smooth( wordTag_initCount[pair], k, num_startin_pos, k_n)
                # EMISSION PROBABILITIES
                else:
                    # IF WORD IS A HAPAX
                    hapax_prob = wordTag_once_count[pair[1]]+k / (words_occurring_once +k * num_Tags)
                    if(word_dict.get(pair[1],-1) != -1):
                         k_n = k * (vocab_size + 1)
                         emit_probability_dict[pair] = laplace_smooth(wordTag_count[pair], k, Tag_count[tag], hapax_prob*k_n)
                    else:
                        #print(wordTag_once_count[pair[1]] / words_occurring_once)
                        k_n = k * (vocab_size + 1)
                        #print(wordTag_count[pair],hapax_prob*k,Tag_count[tag], hapax_prob*k_n, vocab_size, hapax_prob)
                        emit_probability_dict[pair] = (laplace_smooth(wordTag_count[pair], hapax_prob*k, 
                            Tag_count[tag], hapax_prob*k_n))
            i +=1

    # TRANSITION PROBABILITIES
    for tag_prev in Tag_count.keys():
        for tag_cur in Tag_count.keys():
            k_n = k * num_Tags
            tag_pair = (tag_prev,tag_cur)
            trans_probability_dict[tag_pair] = laplace_smooth(Tag_pair_count[tag_pair],k,Tag_count[tag_prev],k_n)


    # test_pair = ('DET','NOUN')
    # print("PROB DET-> NOUN: ", trans_probability_dict[test_pair])
    # print("Count(tag_prev->tag_cur): ",Tag_pair_count[test_pair])
    # print("COUNT(tag_prev): ", Tag_count['DET'])
    # print("NUM_OF_TAGS: ", num_Tags)
    # print("K: ", k)
    # print(math.log(2))

    # test_pair = ('said','VERB')
    # print("PROB count, NOUN: ", init_probability_dict[test_pair])
    # print("Count(tag, start): ",Tag_pair_count[test_pair])
    # print("COUNT(starting pos): ", num_startin_pos)
    # print(k*num_Tags)

    # TRELLIS
    
    # n = 3
    # m = len(Tag_count.keys())
    # trellis = [ [[smallest_num,None] for v in range(m)] for u in range(n) ]
    # pair = ('the','DET')
    # trellis[0][0] = [init_probability_dict.get(pair) * emit_probability_dict.get(pair), None]
    # print(trellis[0][0])
    # raise Exception("BREAK")
    i = 0 # current index of sentence
    j = 0 # current index of Tag_count.keys()
    for sentence in test:
        i = 0
        n = len(sentence)
        m = len(Tag_count.keys())
        trellis = [ [[0,None] for v in range(m)] for u in range(n) ]
        sub_predicts = []
        for word in sentence:
            j = 0
            for tag in Tag_count.keys():
                # STARTING POSITION OF SENTENCE
                pair = tuple((word,tag))
                if(i == 0): # NO parent = None
                    #print(word, " ",tag)
                    #print(init_probability_dict.get(pair, 0) + emit_probability_dict.get(pair,0), " ", word, " ", tag)
                    block = init_probability_dict.get(pair,0) + emit_probability_dict.get(pair,0)
                    trellis[i][j] = [block, None]
                    parent_dict[(i,j)] = (word,tag)
                    #print(trellis[i][j])
                # AFTER STARTING POSITION
                else:
                    transition_list =[]
                    z = 0 # another counter for tag index
                    #print(i-1)
                    #print(trellis[i-1])
                    parent_dict[(i,j)] = (word,tag)
                    for node_value,prev_tag in zip(trellis[i - 1], Tag_count.keys()): # find max transition
                        tag_tup = (prev_tag,tag)
                        transition_list.append(node_value[z] + trans_probability_dict[(tag_tup)])
                    max_trans_prob = max(transition_list)
                    max_trans_idx = np.argmax(transition_list)
                    # print(transition_list)
                    # print(max_trans_idx)
                    # print(max_trans_prob)
                    trellis[i][j] = [emit_probability_dict[pair]+max_trans_prob, (i -1,max_trans_idx)]
                j += 1
            i += 1

        # END OF SENTENCE, END OF INNER FOR LOOP
        # FIND THE MAX TERMINAL NODE AND ASSIGN APPROPRIATE WORD/TAG PAIR
        terminal_node_list = []
        terminal_nodeIdx_list = []
        for node_terminal, tag_2 in zip(trellis[len(sentence)-1],Tag_count.keys()):
            terminal_node_list.append(node_terminal[0])
            terminal_nodeIdx_list.append(node_terminal[1])
        max_terminal_node = max(terminal_node_list) # Get max terminal node probability
        max_terminal_node_idx = np.argmax(terminal_node_list) # index of max node in terminal_node_list
        parent = terminal_nodeIdx_list[max_terminal_node_idx] # parent of max terminal node
        sub_predicts.append( parent_dict[( len(sentence) -1, max_terminal_node_idx)] )
        #print("TRELLIS: ", trellis[len(sentence)-1])
        #print("MAX_NODE ", max_terminal_node, "IDX", max_terminal_node_idx, "PAR",parent)
        
        # TRACE BACKWARDS FROM  TERMINAL NODE
        while(parent!= None):
            temp_i = parent[0]
            temp_j = parent[1]
            parent_node = trellis[temp_i][temp_j]
            word_tag = parent_dict[ (temp_i,temp_j) ]
            sub_predicts.append(word_tag)
            parent = parent_node[1] #get parent of next node
        sub_predicts.reverse()
        #print(sub_predicts)
        predicts.append(sub_predicts)

    #raise Exception("BREAK")
    return predicts


    predicts = []
    raise Exception("You must implement me")
    return predicts