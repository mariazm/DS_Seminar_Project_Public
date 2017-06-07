#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy as np
import random
import sys
import pandas as pd
from collections import defaultdict
from logging import getLogger
from sklearn import linear_model
import math
from collections import Counter

np.random.seed(10)
random.seed(10)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference



class sLDA:
    '''
    Latent Dirichlet Allocation with Collapsed Gibbs Sampling
    '''
    SAMPLING_RATE = 10
    def __init__(self, K, alpha, beta, docs, Y, V, trained=None):
        # set params
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.Y = np.array(Y)
        self.V = V

        # init state
        self._init_state()
        self.trained = trained
        if self.trained is not None:
            self.n_z_w = self.trained.n_z_w
            self.n_z = self.trained.n_z

        # init logger
        self.logger = getLogger(self.__class__.__name__)

    def _init_state(self):
        '''
        Initialize
            - z_m_n: topics assigned to word slots in documents
            - n_m_z: freq. of topics assigned to documents
            - n_z_w: freq. of words assigned to topics
            - n_z:   freq. of topics assigned
        '''
        # assign zero + hyper-params
        self.z_m_n = []
        self.n_m_z = np.zeros((len(self.docs), self.K)) + self.alpha
        self.n_z_w = np.zeros((self.K, self.V)) + self.beta
        self.n_z = np.zeros(self.K) + self.V * self.beta

        # randomly assign topics
        self.N = 0
        for m, doc in enumerate(self.docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                z = np.random.randint(0, self.K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_w[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))


        self.update_z_mean()
        self.update_normal_params()


    def update_z_mean(self, idx=None): 
        if idx == None:
            self.z_mean = [] 
            
            for z_n in self.z_m_n:
                dict_default = Counter(dict(zip(range(self.K),[0]*self.K)))
                cnt = Counter(z_n)
                dict_default.update(cnt)
                aux_k = pd.Series(dict_default).sort_index()
                #aux_k[:] = 0
                #res_z = pd.Series(z_n).value_counts()
                #aux_k =aux_k.add(res_z, fill_value = 0).sort_index()
                z_d_mean = list(aux_k.copy()/len(z_n))
                self.z_mean.append(z_d_mean)
        else:
            dict_default = Counter(dict(zip(range(self.K),[0]*self.K)))
            cnt = Counter(self.z_m_n[idx])
            dict_default.update(cnt)
            aux_k = pd.Series(dict_default).sort_index()
            z_d_mean = list(aux_k.copy()/len(self.z_m_n[idx]))
            self.z_mean[idx] = z_d_mean



    def get_alpha_n_m_z(self, idx=None):
        '''
        Return self.n_m_z (including alpha)
        '''
        if idx is None:
            return self.n_m_z
        else:
            return self.n_m_z[idx]




    def add_last_term(self,n):
        #inte = softmax(np.dot(np.array(self.eta).T, self.Y[n]))
	#r= inte

        #Calculating the term 
	a1 = self.Y[n]
	a2 = self.z_mean[n]
        inte = (self.Y[n]- np.dot(self.eta.T, self.z_mean[n]))**2 / (2* self.sigma2)
        #print inte 
        calc = 1/(2 * math.pi*self.sigma2)**0.5 
        r= calc* np.exp(-inte)
	return r 


    def update_normal_params(self):
        Y = self.Y
        #a_z_mean = np.array(self.z_mean)
        #IF TARGET IS DISCRETE
        #lr =linear_model.LogisticRegression(fit_intercept=False)
        #lr.fit(self.z_mean,self.Y)
        #sq = 

        #USING SKLEARN TO OPTIMIZE TIMES - IF TARGET IS CONTINUOUS
        #print(pd.Series(Y).value_counts())
	lr =linear_model.LinearRegression(fit_intercept=False)
        lr.fit(np.array(self.z_mean),self.Y)
        sq = np.mean((lr.predict(self.z_mean)-self.Y)**2)/len(self.Y)
        self.eta = lr.coef_
        self.sigma2 = sq


    def inference(self):
        '''
        Re-assignment of topics to words
        '''
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            term_extra = self.add_last_term(m)
            #print("document", m)
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                self.discount(z_n, n_m_z, n, t)

                # sampling topic new_z for t
                p_z = self.n_z_w[:, t] * self.get_alpha_n_m_z(m) / self.n_z
                p_z = p_z* term_extra
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                self.assignment(z_n, n_m_z, n, t, new_z)

            #Update normal parameters
            self.update_z_mean(m)
            self.update_normal_params()

    def discount(self, z_n, n_m_z, n, t):
        '''
        Cancel a topic assigned to a word slot
        '''
        z = z_n[n]
        n_m_z[z] -= 1

        if self.trained is None:
            self.n_z_w[z, t] -= 1
            self.n_z[z] -= 1


    def perplexity_new_docs(self, new_docs):
        '''
        Compute the perplexity
        '''
        if self.trained is None:
            phi = self.worddist()
        else:
            phi = self.trained.worddist()
        thetas = self.topicdist()
        log_per = 0
        N = 0
        for m, doc in enumerate(new_docs):
            theta = thetas[m]
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)




    def assignment(self, z_n, n_m_z, n, t, new_z):
        '''
        Assign a topic to a word slot
        '''
        z_n[n] = new_z
        n_m_z[new_z] += 1

        if self.trained is None:
            self.n_z_w[new_z, t] += 1
            self.n_z[new_z] += 1

    def worddist(self):
        '''
        phi = P(w|z): word probability of each topic
        '''
        return self.n_z_w / self.n_z[:, np.newaxis]

    def get_alpha(self):
        '''
        fixed alpha
        '''
        return self.alpha

    def topicdist(self):
        '''
        theta = P(z|d): topic probability of each document
        '''
        doclens = np.array(list(map(len, self.docs)))
        return self.get_alpha_n_m_z()\
            / (doclens[:, np.newaxis] + self.K * self.get_alpha())

    def perplexity(self):
        '''
        Compute the perplexity
        '''
        if self.trained is None:
            phi = self.worddist()
        else:
            phi = self.trained.worddist()
        thetas = self.topicdist()
        log_per = 0
        N = 0
        for m, doc in enumerate(self.docs):
            theta = thetas[m]
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)

    def learning(self, iteration, voca):
        '''
        Repeat inference for learning
        '''
        perp = self.perplexity()
        self.log(self.logger.info, "PERP0", [perp])
        for i in range(iteration):
            self.hyperparameter_learning()
            self.inference()
	    print i 
            if (i + 1) % self.SAMPLING_RATE == 0:
                perp = self.perplexity()
                self.log(self.logger.info, "PERP%s" % (i+1), [perp])
                acc = self.calc_accuracy(self.docs, self.Y)
            	print("Epoch %s, pplxt %s, acc %s" %(i,perp, acc))
        self.output_word_dist_with_voca(voca)

    def hyperparameter_learning(self):
        '''
        No hyperparameter learning in LDA
        '''
        pass


    def calc_accuracy(self,new_docs, y_real):
        predicted = self.predict_y(new_docs)
        error = 0.
        for i in range(len(predicted)):
            y_p =  np.argmax(predicted[i])+1
            if  y_p!= y_real[i]:
                error+= 1
        return 1-(error/len(predicted))

    def predict_y(self, new_docs): 
        prediction = []

        for doc in new_docs:
            z_doc = []
            for n, word in enumerate(doc): 
                p_z = self.n_z_w[:, word] 
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                z_doc.append(new_z)
            #print(len(doc),len(z_doc))
            dict_default = Counter(dict(zip(range(self.K),[0]*self.K)))
            cnt = Counter(z_doc)
            dict_default.update(cnt)
            aux_k = pd.Series(dict_default).sort_index()
            z_d_mean = list(aux_k.copy()/len(doc))
            val = np.dot(self.eta.T, z_d_mean)
            prediction.append(val)
        return prediction


    def word_dist_with_voca(self, voca, topk=None):
        '''
        Output the word probability of each topic
        '''
        phi = self.worddist()
        if topk is None:
            topk = phi.shape[1]
        result = defaultdict(dict)
        for k in range(self.K):
            for w in np.argsort(-phi[k])[:topk]:
                result[k][voca[w]] = phi[k, w]
        return result

    def output_word_dist_with_voca(self, voca, topk=10):
        word_dist = self.word_dist_with_voca(voca, topk)
        for k in word_dist:
            word_dist[k] = sorted(word_dist[k].items(),
                key=lambda x: x[1], reverse=True)
            for w, v in word_dist[k]:
                self.log(self.logger.debug, "TOPIC", [k, w, v])

    def log(self, method, etype, messages):
        method("\t".join(map(str, [self.params(), etype] + messages)))

    def params(self):
        return '''K=%d, alpha=%s, beta=%s''' % (self.K, self.alpha, self.beta)

    def __getstate__(self):
        '''
        logger cannot be serialized
        '''
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        '''
        logger cannot be serialized
        '''
        self.__dict__.update(state)
        self.logger = getLogger(self.__class__.__name__)
