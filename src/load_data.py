import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN

from logistic_sgd import LogisticRegression
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from operator import itemgetter




def load_entailment_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            question=tokens[1].strip().lower().split() #lowercase
            answer=tokens[2].strip().lower().split()   #lowercase
            if len(question)>max_truncate or len(answer)>max_truncate or len(question)< 2 or len(answer)<2:
                continue #skip this pair
            else:
                Y.append(int(tokens[0])) 
                sents=[question, answer]
                #question
                for i in [0,1]:
                    sent=[]
                    words=sents[i]
                    #true_lengths.append(len(words))
                    length=len(words)
                    for word in words:
                        id=word2id.get(word)
                        sent.append(id)

                    Lengths.append(length)
                    left=(maxlength-length)/2
                    right=maxlength-left-length
                    leftPad.append(left)
                    rightPad.append(right)
     
                    sent=[0]*left+sent+[0]*right
                    data.append(sent)
                line_control+=1
#                 if line_control==50000:
#                     break
        read_file.close()
        if len(Lengths)/2 !=len(Y):
            print 'len(Lengths)/2 !=len(Y)'
            exit(0)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad), line_control

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            question=tokens[1].strip().lower().split() 
            answer=tokens[2].strip().lower().split()   
            if len(question)>max_truncate or len(answer)>max_truncate or len(question)< 2 or len(answer)<2:
                continue #skip this pair
            else:
                Y.append(int(tokens[0])) 
                sents=[question, answer]
                for i in [0,1]:
                    sent=[]
                    words=sents[i]
                    #true_lengths.append(len(words))
                    length=len(words)
                    for word in words:
                        id=word2id.get(word)
                        sent.append(id)

                    Lengths.append(length)
                    left=(maxlength-length)/2
                    right=maxlength-left-length
                    leftPad.append(left)
                    rightPad.append(right)
     
                    sent=[0]*left+sent+[0]*right
                    data.append(sent)
                line_control+=1
#                 if line_control==500:
#                     break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) , line_control

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad, train_size=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', train_size
    indices_test, testY, testLengths, testLeftPad, testRightPad, test_size=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', test_size
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')
#         return T.cast(shared_y, 'int32') # for gpu
        #return shared_y


    indices_train=shared_dataset(indices_train)
    indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad, train_size), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad, test_size)]
    return rval, word_ind-1


def load_word2vec_to_init(rand_values, file):

    readFile=open(file, 'r')
    line_count=1
    for line in readFile:
        tokens=line.strip().split()
        rand_values[line_count]=numpy.array(map(float, tokens[1:]))
        line_count+=1                                            
    readFile.close()
    print 'initialization over...'
    return rand_values


def load_SICK_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength, entailment): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            if entailment:
                Y.append(int(tokens[2])) 
            else:
                Y.append(float(tokens[3])) 
            #question
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==500:
            #    break
        read_file.close()
        if len(Lengths)/2 !=len(Y):
            print 'len(Lengths)/2 !=len(Y)'
            exit(0)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            if entailment:
                Y.append(int(tokens[2])) 
            else:
                Y.append(float(tokens[3])) 
            #Y.append(int(tokens[0]))
            for i in [0,1]:
                sent=[]
                words=tokens[i].strip().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right) 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==200:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        #return T.cast(shared_y, 'int64')
        return T.cast(shared_y, 'int64') # 
        #return shared_y
    def shared_dataset_float(data_y, borrow=True):
        return theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX), borrow=borrow)


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)

    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
    
    if entailment:                            
        train_set_y=shared_dataset(trainY)                             
        test_set_y = shared_dataset(testY)
    else:
        train_set_y=shared_dataset_float(trainY)                             
        test_set_y = shared_dataset_float(testY)        
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1

def load_mts_wikiQA(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values

def load_extra_features(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values