#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import lda
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform

## You should have these scripts
import vocabulary as v
import numpy as np
import scipy.stats as st
import operator


def  print_model_stats(model):
	print "Alpha", model.alpha
	print "Beta", model.beta
	print "Docs with words", len(model.docs)
	print len(model.topicdist()[0]), "topics per doc", len(model.topicdist())
	print len(model.worddist()[0]), "word-assignment per topic", len(model.worddist())
	print "Voc size", voca.size()

def get_rank_term_id(model, voca):
    l =len(model.worddist()[0])
    ## To do Spearman we need rank of all words by index
    all_words_ordered = model.word_dist_with_voca(voca, topk=l)
    
    columns_names = ['W_lda_'+str(x) for x in range(l)]
    columns_names.insert(0,'Cluster_lda')

    term_id_ranked = pd.DataFrame(columns=columns_names).T
    
    for i in range(model.K):
        term_list_i =  list(all_words_ordered[i].keys())
        term_to_idx_i = [voca.vocas_id[x] for x in term_list_i]
        term_to_idx_i.insert(0,i)
        term_id_ranked[i] = term_to_idx_i
    return term_id_ranked



def get_spearman_matrix(model1, model2,voca):
    term_matrix1 = get_rank_term_id(model1, voca)
    term_matrix2 = get_rank_term_id(model2, voca)
    
    result = np.zeros((model1.K, model2.K))
    
    for i in range(model1.K):
        for j in range(model2.K):
        	#print(len(term_matrix1[i]), len(term_matrix2[j]))
        	np.put(result, [i,j], st.spearmanr(term_matrix1[i], term_matrix2[j]).correlation)
    return result   

def compute_distances(dataset, distance_measures, n, column = 'index', name=0 ):
    '''
    dataset -- weights of words (same length of columns in each cluster)
    
    name -- name of cluster (number assignment)
    
    distance_measures -- list of [ ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, 
    ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, 
    ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’ ]
    
    n -- number of most similar that we want to get
    
    Example: compute_distance('id_example', ['euclidean'], 6)
    
    '''
    distances = pd.DataFrame()
    from scipy.spatial import distance
    from scipy.spatial.distance import pdist, squareform
    
    ## Find the location (row) - topic - we are looking for
    ## (include "name in parameters if we want two particuar rows of a dataframe)
    if column == 'index':
        id_location = np.where(dataset.index == name)[0][0]
    else:
        id_location = np.where(dataset[column] == name)[0][0]

    # Go through all distance measures we care about
    print n, "' Clusters that are closer to topic =", name
    print "Format: (cluster number, similarity measure)"
    
    for distance_measure in distance_measures:
    
        # Find all pairwise distances
        current_distances = distance.squareform(distance.pdist(dataset, distance_measure))
        # Get the closest n elements for the whiskey we care about
        most_similar = np.argsort(current_distances[:, id_location])[1:n+1]
        # Append results (a new column to the dataframe with the name of the measure)
        distances[distance_measure] = list(zip(dataset.index[most_similar], current_distances[most_similar, id_location]))
        
    return distances

def prepare_data_distances(model, num_words_to_compare, voca):

 	topwords = model.word_dist_with_voca(voca, topk=num_words_to_compare)

	#for i in topwords: print i, "TOP WORDS", topwords[i],"\n"      
	columns_names = ['W_lda_'+str(x) for x in range(num_words_to_compare)]
	columns_names.insert(0,'Cluster_lda')

	weights_topwords = pd.DataFrame(columns=columns_names).T
	weight_words_topwords = pd.DataFrame(columns=columns_names).T
	dataframe_topwords = pd.DataFrame(columns=columns_names).T

	for i in range(model.K):
	    weight_list_i = list(np.sort(topwords[i].values())[::-1])
	    weight_list_i.insert(0,i)
	    weights_topwords[i] = weight_list_i
	    word_list_i = list(np.sort(topwords[i].keys()))
	    word_list_i.insert(0,i)
	    dataframe_topwords[i] = word_list_i
	    word_weights_i = list([ x[0] for x in sorted(topwords[i].items(), key=operator.itemgetter(1),reverse=1) ])
	    word_weights_i.insert(0,i)
	    weight_words_topwords[i] = word_weights_i
	weights_topwords = weights_topwords.T
	dataframe_topwords = dataframe_topwords.T
	weight_words_topwords = weight_words_topwords.T
	return weights_topwords, dataframe_topwords, weight_words_topwords


def get_matrix_distances(model1, model2, num_words_to_compare, voca, distance_measures): 
	weight_matrix1,_,_ = prepare_data_distances(model1, num_words_to_compare, voca)
	weight_matrix2,_,_ = prepare_data_distances(model2, num_words_to_compare, voca)
	for i in range(model1.K):
		print("Comparing model 1, topic", i)
		dataset = weight_matrix2.copy()
		inp = weight_matrix1.copy().iloc[i]
		inp['Cluster_lda'] = '-1'
		dataset.loc[-1] = inp
		print compute_distances(dataset, distance_measures, n=5, column = 'Cluster_lda', name=-1)

	for i in range(model2.K):
		print("Comparing model 2, topic", i)
		dataset = weight_matrix1.copy()
		inp = weight_matrix2.copy().iloc[i]
		inp['Cluster_lda'] = '-1'
		dataset.loc[-1] = inp
		print compute_distances(dataset, distance_measures, n=5, column = 'Cluster_lda', name=-1)







