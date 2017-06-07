#!/usr/bin/env python
# -*- coding: utf-8 -*-


import gzip
import html 
import json
import os
import pandas as pd
import numpy as np
import pickle 
import lda
import slda
import dmr
from nltk.corpus import stopwords
import re
import random
import optparse
import vocabulary as v

parser = optparse.OptionParser()

parser.add_option('--epochs', '--epochs', action="store", dest="epochs", help="Total number of epochs", default=4000)
parser.add_option('--dataroot', '--dataroot', action="store", dest="dataroot", help="Root ", default="./../")
parser.add_option('--alpha', '--alpha', action="store", dest="alpha", help="Alpha value ", default=0.1)
parser.add_option('--beta', '--beta', action="store", dest="beta", help="Beta value ", default=0.01)
parser.add_option('--topics', '--topics', action="store", dest="topics", help="K: number of topics ", default=5.)
parser.add_option('--model', '--model', action="store", dest="model", help="Launch model: lda,slda, dmr ", default= 'lda')
parser.add_option('--savemodel', '--savemodel', action = "store", dest = "savemodel", help = "Name of file to be saved" , default = "model")
np.random.seed(10)
random.seed(10)

options, args = parser.parse_args()
print("MODELS")
print("Parameters:")
print(options)

#Reading sample 
sample = pd.read_pickle(options.dataroot)
print "Sample read! Original rows ", len(sample)

##### CLEANING 

uni_length = [len(i) for i in sample.text]
sample['aux_len'] =uni_length
sample_final = sample[sample.aux_len >2]
sample_final = sample_final.drop('aux_len', axis =1)
sample_final = sample_final.reset_index()
	
print "Cleaned sample! Final rows ",len(sample_final)
#######SPLITING TRAIN TEST

rows = random.sample(sample_final.index, 2000)
sample_train = sample_final.drop(rows)
sample_test = sample_final.ix[rows]

print "Final test and train sizes:", len(sample_test), len(sample_train)
cleaned_reviews_train = sample_train['text']
cleaned_reviews_test = sample_test['text']
print("Sample cleaned!")


############ SETTING THE VARIABLES

#Text
voca = v.Vocabulary()
docs = voca.read_corpus(cleaned_reviews_train)
docs_test = voca.new_corpus(cleaned_reviews_test)

if options.model == 'slda':
    # Supervised
    Y_train = sample_train.stars
if options.model == 'dmr':
    feat_orig = np.reshape(sample_train.stars, (len(sample_train.stars),1))
    #Features
    sample_train.columns = [s.encode('utf-8') for s in sample_train.columns]
    features_biz = sample_train.filter(regex='biz_')
    features_biz = features_biz.drop('biz_name', axis =1)
    #veeecs = features_biz
    #vecs = np.array([[v for v in vec] for vec in vecs], dtype=np.float32)
    feat_biz  =np.array(features_biz) 


	    #Features
    #sample_train.columns = [s.encode('utf-8') for s in sample_train.columns]
    features_usr = sample_train.filter(regex='usr_')
    #features_usr = features_biz.drop('biz_name', axis =1)
    #vecs = features_usr
    #vecs = np.array([[v for v in vec] for vec in vecs], dtype=np.float32)
    feat_usr  = np.array(features_usr)


filename = options.savemodel
############ LDA 
if options.model == 'lda':
    print("Starting LDA")
    lda = lda.LDA(int(options.topics), options.alpha, options.beta, docs, voca.size())
    lda.learning(int(options.epochs),voca)
    pickle.dump(lda, open(filename+'_lda.p', 'wb'))
    pplt_test = lda.perplexity_new_docs(docs_test)
    print "Test perplexity: ", pplt_test
    print("LDA, LEARNING FINISHED, MODEL SAVED")

###########sLDA
elif options.model == 'slda':
    print("Starting sLDA")
    slda = slda.sLDA(int(options.topics), options.alpha, options.beta, docs,Y_train, voca.size())
    slda.learning(int(options.epochs),voca)
    pickle.dump(slda, open(filename+'_slda.p', 'wb'))
    print("Test perplexity: ",slda.perplexity_new_docs(docs_test))
    print("sLDA, LEARNING FINISHED, MODEL SAVED")

############### DMR 
elif options.model== 'dmr': 
    print("Starting DMR models ")
    dmr1 = dmr.DMR(int(options.topics), options.alpha, options.beta, docs,feat_orig, voca.size())
    dmr1.learning(int(options.epochs),voca)
    pickle.dump(dmr1, open(filename+'_dmr_y.p', 'wb'))
    print("Test perplexity: ",dmr1.perplexity_new_docs(docs_test))
    print("DMR JUST Y, LEARNING FINISHED, MODEL SAVED")

    dmr2 = dmr.DMR(int(options.topics), options.alpha, options.beta, docs,feat_biz, voca.size())
    dmr2.learning(int(options.epochs),voca)
    pickle.dump(dmr2, open(filename+'_dmr_biz.p', 'wb'))
    print("Test perplexity: ",dmr2.perplexity_new_docs(docs_test))
    print("DMR BIZ, LEARNING FINISHED, MODEL SAVED")

    dmr3 = dmr.DMR(int(options.topics), options.alpha, options.beta, docs,feat_usr, voca.size())
    dmr3.learning(int(options.epochs),voca)
    pickle.dump(dmr3, open(filename+'_dmr_user.p', 'wb'))
    print("Test perplexity: ",dmr3.perplexity_new_docs(docs_test))
    print("DMR USER, LEARNING FINISHED, MODEL SAVED")

