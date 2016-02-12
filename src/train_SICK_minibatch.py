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
import time

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from load_data import load_entailment_corpus , load_word2vec_to_init, load_SICK_corpus, load_mts_wikiQA, load_extra_features
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Average_Pooling_for_Top, create_conv_para, create_GRU_para, GRU_Tensor3_Input, GRU_Matrix_Input, Matrix_Bit_Shift, GRU_Batch_Tensor_Input, compute_simi_feature_matrix_with_matrix
from random import shuffle

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import linalg, mat, dot



#for gpu, we need change the load data function, Dim_Align, and lscalar, dmatrix blabla

def evaluate_lenet5(learning_rate=0.0001, n_epochs=2000, nkerns=[50,50], batch_size=10, window_width=3,
                    maxSentLength=64, emb_size=50, hidden_size=200,
                    margin=0.5, L2_weight=0.0006, update_freq=1, norm_threshold=5.0, max_truncate=33):# max_truncate can be 45
    maxSentLength=max_truncate+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SICK/';
    rng = numpy.random.RandomState(23455)
#     datasets, vocab_size=load_SICK_corpus(rootPath+'vocab_lower_in_word2vec.txt', rootPath+'train.txt', rootPath+'test.txt', max_truncate,maxSentLength)#vocab_size contain train, dev and test
    datasets, vocab_size=load_SICK_corpus(rootPath+'vocab.txt', rootPath+'train_plus_dev.txt', rootPath+'test.txt', max_truncate,maxSentLength, entailment=True)
    mt_train, mt_test=load_mts_wikiQA(rootPath+'Train_plus_dev_MT/concate_14mt_train.txt', rootPath+'Test_MT/concate_14mt_test.txt')
    extra_train, extra_test=load_extra_features(rootPath+'train_plus_dev_rule_features_cosine_eucli_negation_len1_len2_syn_hyper1_hyper2_anto(newsimi0.4).txt', rootPath+'test_rule_features_cosine_eucli_negation_len1_len2_syn_hyper1_hyper2_anto(newsimi0.4).txt')
    discri_train, discri_test=load_extra_features(rootPath+'train_plus_dev_discri_features_0.3.txt', rootPath+'test_discri_features_0.3.txt')
    
    
    indices_train, trainY, trainLengths, normalized_train_length, trainLeftPad, trainRightPad= datasets[0]
    indices_train_l=indices_train[::2,:]
    indices_train_r=indices_train[1::2,:]
    trainLengths_l=trainLengths[::2]
    trainLengths_r=trainLengths[1::2]
    normalized_train_length_l=normalized_train_length[::2]
    normalized_train_length_r=normalized_train_length[1::2]

    trainLeftPad_l=trainLeftPad[::2]
    trainLeftPad_r=trainLeftPad[1::2]
    trainRightPad_l=trainRightPad[::2]
    trainRightPad_r=trainRightPad[1::2]    
    
    
    indices_test, testY, testLengths,normalized_test_length, testLeftPad, testRightPad = datasets[1]
    indices_test_l=indices_test[::2,:]
    indices_test_r=indices_test[1::2,:]
    testLengths_l=testLengths[::2]
    testLengths_r=testLengths[1::2]
    normalized_test_length_l=normalized_test_length[::2]
    normalized_test_length_r=normalized_test_length[1::2]
    
    testLeftPad_l=testLeftPad[::2]
    testLeftPad_r=testLeftPad[1::2]
    testRightPad_l=testRightPad[::2]
    testRightPad_r=testRightPad[1::2]  

    n_train_batches=indices_train_l.shape[0]/batch_size
    n_test_batches=indices_test_l.shape[0]/batch_size
    
    train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
    test_batch_start=list(numpy.arange(n_test_batches)*batch_size)

    
    indices_train_l=theano.shared(numpy.asarray(indices_train_l, dtype=theano.config.floatX), borrow=True)
    indices_train_r=theano.shared(numpy.asarray(indices_train_r, dtype=theano.config.floatX), borrow=True)
    indices_test_l=theano.shared(numpy.asarray(indices_test_l, dtype=theano.config.floatX), borrow=True)
    indices_test_r=theano.shared(numpy.asarray(indices_test_r, dtype=theano.config.floatX), borrow=True)
    indices_train_l=T.cast(indices_train_l, 'int64')
    indices_train_r=T.cast(indices_train_r, 'int64')
    indices_test_l=T.cast(indices_test_l, 'int64')
    indices_test_r=T.cast(indices_test_r, 'int64')
    '''
    indices_train_l=T.cast(indices_train_l, 'int32')
    indices_train_r=T.cast(indices_train_r, 'int32')
    indices_test_l=T.cast(indices_test_l, 'int32')
    indices_test_r=T.cast(indices_test_r, 'int32')
    '''


    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    #rand_values[0]=numpy.array([1e-50]*emb_size)
    rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_glove_50d.txt')
#     rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_lower_in_word2vec_embs_300d.txt')
    embeddings=theano.shared(value=rand_values, borrow=True)      
    

    error_sum=0
    
    # allocate symbolic variables for the data
    index = T.lscalar()
    x_index_l = T.lmatrix('x_index_l')   # now, x is the index matrix, must be integer
    x_index_r = T.lmatrix('x_index_r')
    y = T.lvector('y')  
    left_l=T.lvector()
    right_l=T.lvector()
    left_r=T.lvector()
    right_r=T.lvector()
    length_l=T.lvector()
    length_r=T.lvector()
    norm_length_l=T.dvector()
    norm_length_r=T.dvector()
    mts=T.dmatrix()
    extra=T.dmatrix()
    discri=T.dmatrix()
    cost_tmp=T.dscalar()




#     #GPU
#     index = T.iscalar()
#     x_index_l = T.imatrix('x_index_l')   # now, x is the index matrix, must be integer
#     x_index_r = T.imatrix('x_index_r')
#     y = T.ivector('y')  
#     left_l=T.iscalar()
#     right_l=T.iscalar()
#     left_r=T.iscalar()
#     right_r=T.iscalar()
#     length_l=T.iscalar()
#     length_r=T.iscalar()
#     norm_length_l=T.fscalar()
#     norm_length_r=T.fscalar()
#     #mts=T.dmatrix()
#     #wmf=T.dmatrix()
#     cost_tmp=T.fscalar()
    #x=embeddings[x_index.flatten()].reshape(((batch_size*4),maxSentLength, emb_size)).transpose(0, 2, 1).flatten()
    ishape = (emb_size, maxSentLength)  # this is the size of MNIST images
    filter_size=(emb_size,window_width)
    #poolsize1=(1, ishape[1]-filter_size[1]+1) #?????????????????????????????
    length_after_wideConv=ishape[1]+filter_size[1]-1
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #layer0_input = x.reshape(((batch_size*4), 1, ishape[0], ishape[1]))
    layer0_l_input = debug_print(embeddings[x_index_l.flatten()].reshape((batch_size, maxSentLength, emb_size)).transpose(0,2,1), 'layer0_l_input')
    layer0_r_input = debug_print(embeddings[x_index_r.flatten()].reshape((batch_size, maxSentLength, emb_size)).transpose(0,2,1), 'layer0_r_input')
    #paras:
    U, W, b=create_GRU_para(rng, emb_size, nkerns[0])
    layer0_para=[U, W, b]     
    U1, W1, b1=create_GRU_para(rng, nkerns[0], nkerns[1])
    layer1_para=[U1, W1, b1] 
    def loop (l_left, l_right, l_matrix, r_left, r_right, r_matrix, mts_i, extra_i, norm_length_l_i, norm_length_r_i):   
        l_input_tensor=debug_print(Matrix_Bit_Shift(l_matrix[:,l_left:-l_right]), 'l_input_tensor')
        r_input_tensor=debug_print(Matrix_Bit_Shift(r_matrix[:,r_left:-r_right]), 'r_input_tensor')
        
        addition_l=T.sum(l_matrix[:,l_left:-l_right], axis=1)
        addition_r=T.sum(r_matrix[:,r_left:-r_right], axis=1)
        cosine_addition=cosine(addition_l, addition_r)
        eucli_addition=1.0/(1.0+EUCLID(addition_l, addition_r))#25.2%
        
        layer0_A1 = GRU_Batch_Tensor_Input(X=l_input_tensor, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)
        layer0_A2 = GRU_Batch_Tensor_Input(X=r_input_tensor, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)
        
        cosine_sent=cosine(layer0_A1.output_sent_rep, layer0_A2.output_sent_rep)
        eucli_sent=1.0/(1.0+EUCLID(layer0_A1.output_sent_rep, layer0_A2.output_sent_rep))#25.2%
        
        attention_matrix=compute_simi_feature_matrix_with_matrix(layer0_A1.output_matrix, layer0_A2.output_matrix, layer0_A1.dim, layer0_A2.dim, maxSentLength*(maxSentLength+1)/2)
        
        l_max_attention=T.max(attention_matrix, axis=1)
        neighborsArgSorted = T.argsort(l_max_attention)
        kNeighborsArg = neighborsArgSorted[:3]#only average the min 3 vectors
        ll = T.sort(kNeighborsArg).flatten() # make y indices in acending lie
    
    
        r_max_attention=T.max(attention_matrix, axis=0)
        neighborsArgSorted_r = T.argsort(r_max_attention)
        kNeighborsArg_r = neighborsArgSorted_r[:3]#only average the min 3 vectors
        rr = T.sort(kNeighborsArg_r).flatten() # make y indices in acending lie
    
        
        l_max_min_attention=debug_print(layer0_A1.output_matrix[:,ll], 'l_max_min_attention')
        r_max_min_attention=debug_print(layer0_A2.output_matrix[:,rr], 'r_max_min_attention')
        

    
        layer1_A1=GRU_Matrix_Input(X=l_max_min_attention, word_dim=nkerns[0], hidden_dim=nkerns[1],U=U1,W=W1,b=b1,bptt_truncate=-1)
        layer1_A2=GRU_Matrix_Input(X=r_max_min_attention, word_dim=nkerns[0], hidden_dim=nkerns[1],U=U1,W=W1,b=b1,bptt_truncate=-1)
    
        vec_l=debug_print(layer1_A1.output_vector_last.reshape((1, nkerns[1])), 'vec_l')
        vec_r=debug_print(layer1_A2.output_vector_last.reshape((1, nkerns[1])), 'vec_r')
    
        
        
    #     sum_uni_l=T.sum(layer0_l_input, axis=3).reshape((1, emb_size))
    #     aver_uni_l=sum_uni_l/layer0_l_input.shape[3]
    #     norm_uni_l=sum_uni_l/T.sqrt((sum_uni_l**2).sum())
    #     sum_uni_r=T.sum(layer0_r_input, axis=3).reshape((1, emb_size))
    #     aver_uni_r=sum_uni_r/layer0_r_input.shape[3]
    #     norm_uni_r=sum_uni_r/T.sqrt((sum_uni_r**2).sum())
    #     
        uni_cosine=cosine(vec_l, vec_r)
    #     aver_uni_cosine=cosine(aver_uni_l, aver_uni_r)
    #     uni_sigmoid_simi=debug_print(T.nnet.sigmoid(T.dot(norm_uni_l, norm_uni_r.T)).reshape((1,1)),'uni_sigmoid_simi')    
    #     '''
    #     linear=Linear(sum_uni_l, sum_uni_r)
    #     poly=Poly(sum_uni_l, sum_uni_r)
    #     sigmoid=Sigmoid(sum_uni_l, sum_uni_r)
    #     rbf=RBF(sum_uni_l, sum_uni_r)
    #     gesd=GESD(sum_uni_l, sum_uni_r)
    #     '''
        eucli_1=1.0/(1.0+EUCLID(vec_l, vec_r))#25.2%
    #     #eucli_1_exp=1.0/T.exp(EUCLID(sum_uni_l, sum_uni_r))
    #     
        len_l=norm_length_l_i.reshape((1,1))
        len_r=norm_length_r_i.reshape((1,1))  
    #     
    #     '''
    #     len_l=length_l.reshape((1,1))
    #     len_r=length_r.reshape((1,1))  
    #     '''
        #length_gap=T.log(1+(T.sqrt((len_l-len_r)**2))).reshape((1,1))
        #length_gap=T.sqrt((len_l-len_r)**2)
        #layer3_input=mts
#         layer3_input_nn=T.concatenate([vec_l, vec_r,
#                                     cosine_addition, eucli_addition,
#     #                                 cosine_sent, eucli_sent,
#                                     uni_cosine,eucli_1], axis=1)#, layer2.output, layer1.output_cosine], axis=1)
        
        output_i=T.concatenate([vec_l, vec_r,
                                    cosine_addition, eucli_addition,
    #                                 cosine_sent, eucli_sent,
                                    uni_cosine,eucli_1,
                                    mts_i.reshape((1,14)),
                                    len_l, len_r,
                                    extra_i.reshape((1,9))], axis=1)#, layer2.output, layer1.output_cosine], axis=1)    
        return output_i
    
    layer3_input, _ = theano.scan(fn=loop,
                            sequences=[left_l, right_l, layer0_l_input, left_r, right_r, layer0_r_input, mts, extra, norm_length_l, norm_length_r],
                            outputs_info=None,#[self.h0, None],
                            n_steps=batch_size)       
#l_left, l_right, l_matrix, r_left, r_right, r_matrix, mts_i, extra_i, norm_length_l_i, norm_length_r_i
#     x_index_l = T.lmatrix('x_index_l')   # now, x is the index matrix, must be integer
#     x_index_r = T.lmatrix('x_index_r')
#     y = T.lvector('y')  
#     left_l=T.lvector()
#     right_l=T.lvector()
#     left_r=T.lvector()
#     right_r=T.lvector()
#     length_l=T.lvector()
#     length_r=T.lvector()
#     norm_length_l=T.dvector()
#     norm_length_r=T.dvector()
#     mts=T.dmatrix()
#     extra=T.dmatrix()
#     discri=T.dmatrix()
#     cost_tmp=T.dscalar()

    
    #layer3_input=T.concatenate([mts,eucli, uni_cosine, len_l, len_r, norm_uni_l-(norm_uni_l+norm_uni_r)/2], axis=1)
    #layer3=LogisticRegression(rng, input=layer3_input, n_in=11, n_out=2)
    feature_size=2*nkerns[1]+2+2+14+2+9
    layer3_input=layer3_input.reshape((batch_size, feature_size))
    layer3=LogisticRegression(rng, input=layer3_input, n_in=feature_size, n_out=3)

    
    #L2_reg =(layer3.W** 2).sum()+(layer2.W** 2).sum()+(layer1.W** 2).sum()+(conv_W** 2).sum()
    L2_reg =debug_print((layer3.W** 2).sum()+(U** 2).sum()+(W** 2).sum()+(U1** 2).sum()+(W1** 2).sum(), 'L2_reg')#+(layer1.W** 2).sum()++(embeddings**2).sum()
    cost_this =debug_print(layer3.negative_log_likelihood(y), 'cost_this')#+L2_weight*L2_reg
    cost=debug_print((cost_this+cost_tmp)/update_freq+L2_weight*L2_reg, 'cost')
    #cost=debug_print((cost_this+cost_tmp)/update_freq, 'cost')
    

    
    test_model = theano.function([index], [layer3.errors(y),layer3_input, y],
          givens={
            x_index_l: indices_test_l[index: index + batch_size],
            x_index_r: indices_test_r[index: index + batch_size],
            y: testY[index: index + batch_size],
            left_l: testLeftPad_l[index: index + batch_size],
            right_l: testRightPad_l[index: index + batch_size],
            left_r: testLeftPad_r[index: index + batch_size],
            right_r: testRightPad_r[index: index + batch_size],
            length_l: testLengths_l[index: index + batch_size],
            length_r: testLengths_r[index: index + batch_size],
            norm_length_l: normalized_test_length_l[index: index + batch_size],
            norm_length_r: normalized_test_length_r[index: index + batch_size],
            mts: mt_test[index: index + batch_size],
            extra: extra_test[index: index + batch_size],
            discri:discri_test[index: index + batch_size]
            }, on_unused_input='ignore', allow_input_downcast=True)


    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]
    params = layer3.params+ layer1_para+layer0_para#+[embeddings]# + layer1.params 
#     params_conv = [conv_W, conv_b]
    
#     accumulator=[]
#     for para_i in params:
#         eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
#         accumulator.append(theano.shared(eps_p, borrow=True))
#       
#     # create a list of gradients for all model parameters
#     grads = T.grad(cost, params)
# 
#     updates = []
#     for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         grad_i=debug_print(grad_i,'grad_i')
#         acc = acc_i + T.sqr(grad_i)
#         updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
#         updates.append((acc_i, acc))    

    def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(numpy.float64(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates
    
    updates=Adam(cost=cost, params=params, lr=learning_rate)
  
    train_model = theano.function([index,cost_tmp], cost, updates=updates,
          givens={
            x_index_l: indices_train_l[index: index + batch_size],
            x_index_r: indices_train_r[index: index + batch_size],
            y: trainY[index: index + batch_size],
            left_l: trainLeftPad_l[index: index + batch_size],
            right_l: trainRightPad_l[index: index + batch_size],
            left_r: trainLeftPad_r[index: index + batch_size],
            right_r: trainRightPad_r[index: index + batch_size],
            length_l: trainLengths_l[index: index + batch_size],
            length_r: trainLengths_r[index: index + batch_size],
            norm_length_l: normalized_train_length_l[index: index + batch_size],
            norm_length_r: normalized_train_length_r[index: index + batch_size],
            mts: mt_train[index: index + batch_size],
            extra: extra_train[index: index + batch_size],
            discri:discri_train[index: index + batch_size]
            }, on_unused_input='ignore', allow_input_downcast=True)

    train_model_predict = theano.function([index, cost_tmp], [cost_this,layer3.errors(y), layer3_input, y],
          givens={
            x_index_l: indices_train_l[index: index + batch_size],
            x_index_r: indices_train_r[index: index + batch_size],
            y: trainY[index: index + batch_size],
            left_l: trainLeftPad_l[index: index + batch_size],
            right_l: trainRightPad_l[index: index + batch_size],
            left_r: trainLeftPad_r[index: index + batch_size],
            right_r: trainRightPad_r[index: index + batch_size],
            length_l: trainLengths_l[index: index + batch_size],
            length_r: trainLengths_r[index: index + batch_size],
            norm_length_l: normalized_train_length_l[index: index + batch_size],
            norm_length_r: normalized_train_length_r[index: index + batch_size],
            mts: mt_train[index: index + batch_size],
            extra: extra_train[index: index + batch_size],
            discri:discri_train[index: index + batch_size]
            }, on_unused_input='ignore', allow_input_downcast=True)



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.time()

    mid_time = start_time

    epoch = 0
    done_looping = False
    
    acc_max=0.0
    best_epoch=0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
#         shuffle(train_batch_start)#shuffle training data
        cost_tmp=0.0
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
#             if (batch_start+1)%1000==0:
#                 print batch_start+1,  'uses ', (time.time()-mid_time)/60.0, 'min'
            iter = (epoch - 1) * n_train_batches + minibatch_index +1

            minibatch_index=minibatch_index+1
            #if epoch %2 ==0:
            #    batch_start=batch_start+remain_train
            #time.sleep(0.5)
            #print batch_start
            if iter%update_freq != 0:
                cost_ij, error_ij, layer3_input, y=train_model_predict(batch_start, 0.0)
                #print 'layer3_input', layer3_input
                cost_tmp+=cost_ij
                error_sum+=error_ij
                #print 'cost_acc ',cost_acc
                #print 'cost_ij ', cost_ij
                #print 'cost_tmp before update',cost_tmp
            else:
                cost_average= train_model(batch_start,cost_tmp)
                #print 'layer3_input', layer3_input
                error_sum=0
                cost_tmp=0.0#reset for the next batch
                #print 'cost_average ', cost_average
                #print 'cost_this ',cost_this
                #exit(0)
            #exit(0)
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_average)+' error: '+str(error_sum)+'/'+str(update_freq)+' error rate: '+str(error_sum*1.0/update_freq)
            #if iter ==1:
            #    exit(0)
            
            if iter % validation_frequency == 0:
                #write_file=open('log.txt', 'w')
                test_losses=[]
                test_y=[]
                test_features=[]
                for i in test_batch_start:
                    test_loss, layer3_input, y=test_model(i)
                    #test_losses = [test_model(i) for i in test_batch_start]
                    test_losses.append(test_loss)
                    test_y.append(y)
                    test_features.append(layer3_input)
                    #write_file.write(str(pred_y[0])+'\n')#+'\t'+str(testY[i].eval())+
 
                #write_file.close()
                test_score = numpy.mean(test_losses)
                test_features=numpy.concatenate(test_features, axis=0)
                test_y=numpy.concatenate(test_y, axis=0)
                print(('\t\t\t\t\t\tepoch %i, minibatch %i/%i, test acc of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,
                           (1-test_score) * 100.))
                acc_nn=1-test_score
                #now, see the results of LR
                #write_feature=open(rootPath+'feature_check.txt', 'w')
                 
                #this step is risky: if the training data is too big, then this step will make the training time twice longer
                train_y=[]
                train_features=[]
                count=0
                for batch_start in train_batch_start: 
                    cost_ij, error_ij, layer3_input, y=train_model_predict(batch_start, 0.0)
                    train_y.append(y)
                    train_features.append(layer3_input)
                    #write_feature.write(str(batch_start)+' '+' '.join(map(str,layer3_input[0]))+'\n')
                    #count+=1
 
                train_features=numpy.concatenate(train_features, axis=0)
                train_y=numpy.concatenate(train_y, axis=0)
 
                clf = svm.SVC(C=1.0, kernel='linear')
                clf.fit(train_features, train_y)
                results=clf.predict(test_features)
                lr=linear_model.LogisticRegression(C=1e5)
                lr.fit(train_features, train_y)
                results_lr=lr.predict(test_features)
                corr_count=0
                corr_count_lr=0
                test_size=len(test_y)
                for i in range(test_size):
                    if results[i]==test_y[i]:
                        corr_count+=1
                    if results_lr[i]==test_y[i]:
                        corr_count_lr+=1
                acc_svm=corr_count*1.0/test_size
                acc_lr=corr_count_lr*1.0/test_size
                if acc_svm > acc_max:
                    acc_max=acc_svm
                    best_epoch=epoch
                if acc_lr > acc_max:
                    acc_max=acc_lr
                    best_epoch=epoch
                if acc_nn > acc_max:
                    acc_max=acc_nn
                    best_epoch=epoch
                print  'acc_nn:', acc_nn, 'acc_lr:', acc_lr, 'acc_svm:', acc_svm, ' max acc: ',    acc_max , ' at epoch: ', best_epoch  

            if patience <= iter:
                done_looping = True
                break
        
        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()
            
        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def store_model_to_file(best_params):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/snli_1.0//Best_Conv_Para', 'wb')  # this will overwrite current contents
    for para in best_params:           
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi.reshape((1,1))    
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))    
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))    
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))    
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))    
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))   
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20).reshape((1,1))
    


if __name__ == '__main__':
    evaluate_lenet5()