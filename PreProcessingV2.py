

# # 1. Getting the data

import gzip
import html5lib 
import json
import os
import pandas as pd
import numpy as np
import math

from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import coo_matrix, hstack
import pickle 


### Parameters ###
#stopw = set(stopwords.words('english'))
np.random.seed(10)
dir_yelp = 'DataProject/yelp_dataset_challenge_round9/'
output_path = 'DataProject/'
sample_size = 10000
stopw = set(pd.read_pickle(output_path+'stop_words_nltk.p'))



# Listing all the JSON files in the directory
y_dirs = os.listdir(dir_yelp)

#Output dir 
#if not os.path.exists(out_dir):
#    os.makedirs(out_dir)


#############################################################################################################
################################ 1. Setting the structures of the information ###############################
#############################################################################################################

### This part of the code is based on Prof. Perry's code  
### http://ptrckprry.com/course/ssd/data/yelp-academic/01_make_json.py

tables = {}

tables['user'] = {
    'fields': ['user_id', 'name', 'yelping_since', 'average_stars'],
    'subfields': {}
    }

tables['review'] = {
    'fields': ['review_id', 'business_id', 'user_id', 'date', 'stars','text'],
    'subfields': {}
    }

tables['business'] = {
    'fields': ['business_id','name', 'stars', 'is_open','categories', 'state', 'city', 
               'longitude','latitude', 'neighborhood'],
    'subfields': {}
    }


#### Formatting the data in Dataframes


def unescape(x):
    if isinstance(x, str):
        x = html.unescape(x)
    return x

reviews_ext= []
user_ext = []
business_ext = []

for y_dir in y_dirs:
    #The files we're interested in extracting end with ".json"
    r1 = int('user' in  y_dir)
    r2 = int('review' in y_dir )
    r3 = int('business' in y_dir )
    if y_dir.endswith('.json') and (r1==1 or r2==1 or r3==1)  :
        print dir_yelp + y_dir
        with open(dir_yelp+ y_dir, 'r') as dataset:
            #Extracting the lines and formatting accordingly
            for line in dataset:
                obj = json.loads(line.decode('utf-8'))
                tab = tables[obj['type']]
                record = {}
                #print(line)
                for i in tab['fields']:
                    #print(i)
                    if i in tab['subfields']:
                        record[i] = {}
                        for j in tab['subfields'][i]:
                            record[i][j] = unescape(obj[i][j])
                    else:
                        record[i] = unescape(obj[i])
                if 'review_id' in record: #IS REVIEW
                    reviews_ext.append(record)
                if 'yelping_since' in record: #IS USER
                    user_ext.append(record)
                if 'neighborhood' in record:  #IS BUSINESS
                    business_ext.append(record)
                    
print("Reading ready..!")


# Create dataframes

users = pd.DataFrame(user_ext)
businesses = pd.DataFrame(business_ext)
reviews = pd.DataFrame(reviews_ext)


print("Pre-processing..")

############################## Users' Dataset  ##################################################################

name_datetime_variable = 'yelping_since'
users[name_datetime_variable] = pd.to_datetime(users[name_datetime_variable])
users[name_datetime_variable+'_year'] = pd.DatetimeIndex(users[name_datetime_variable]).year
users[name_datetime_variable+'_month'] = pd.DatetimeIndex(users[name_datetime_variable]).month
users[name_datetime_variable+'_day'] = pd.DatetimeIndex(users[name_datetime_variable]).day

# User's Variables

user_variables = ['user_id','average_stars','yelping_since_year']


############################### Business' Dataset ##############################################################


dummies_cities = pd.get_dummies(businesses['state'], dummy_na=False, prefix='state', prefix_sep='_')
businesses_dumm = pd.concat([businesses, dummies_cities], axis=1)

dummies_states = pd.get_dummies(businesses['city'], dummy_na=False, prefix='city', prefix_sep='_')
businesses_dumm = pd.concat([businesses_dumm, dummies_states], axis=1)

dummies_neighborhood = pd.get_dummies(businesses['neighborhood'], dummy_na=False, prefix='neighborhood', prefix_sep='_')
businesses_dumm = pd.concat([businesses_dumm, dummies_neighborhood], axis=1)

businesses_dumm = businesses_dumm.drop(['city','state','neighborhood'],axis=1)


dummies_categories = pd.get_dummies(businesses['categories'].apply(pd.Series).stack()).sum(level=0)
prefix = 'catBiz_'
new_columns = []
for i in dummies_categories.columns:
    new_columns.append( prefix+i )
 
dummies_categories.columns = new_columns   
businesses_sparse = pd.concat([businesses_dumm, dummies_categories], axis=1)

businesses_sparse = businesses_sparse.drop(['categories'],axis=1)

# Business' variables

business_variables = []
for i in businesses_sparse.columns:
    if 'catBiz' in i or 'neighborhood' in i or 'state' in i or 'city' in i or 'name' in i  \
       or 'latitude' in i or 'longitude' in i or 'stars' in i or 'is_open' in i or 'id' in i:
        business_variables.append(i)


############################## Reviews' Dataset  ################################################################

name_datetime_variable = 'date'
reviews[name_datetime_variable] = pd.to_datetime(reviews[name_datetime_variable])
reviews[name_datetime_variable+'_year'] = pd.DatetimeIndex(reviews[name_datetime_variable]).year
reviews[name_datetime_variable+'_month'] = pd.DatetimeIndex(reviews[name_datetime_variable]).month
reviews[name_datetime_variable+'_day'] = pd.DatetimeIndex(reviews[name_datetime_variable]).day

# Review's variables

reviews_variables = ['business_id','user_id','date_year','stars','text']



#####################################################################################################################
################################### 2. Create a sample ##############################################################
#####################################################################################################################

print("Creating samples..")

##################### Functions to join metadata to comments

def create_yelp_columns(join_data):
	
	## Column 1
	join_data['usr_year_experience'] = join_data['date_year']-join_data['usr_yelping_since_year']

    ## Column 2
	join_data['usr_average_stars'] = join_data['usr_average_stars'].map(lambda x: 0.5 * math.ceil(2.0 * round(x,1)))
	join_data['usr_stars_bias']= join_data['usr_average_stars']-join_data['biz_stars']

	join_data = join_data.drop(['usr_yelping_since_year','biz_stars','usr_average_stars'],axis=1)

	return join_data


def join_data(sample_dataset):

    ### 2.1. Join: Reviews and user's info

    user_data_selected = users[user_variables]
    user_data_selected.columns = [ "usr_"+col if col != "user_id" else col for col in user_data_selected.columns ]
    review_user = pd.merge(sample_dataset, user_data_selected, on = "user_id", how = "left")
    
    ### 2.2. Join: Reviews and business info 
    
    business_data_selected = businesses_sparse[business_variables]
    business_data_selected.columns = [ "biz_"+col if col != "business_id" else col for col in business_data_selected.columns ]
    review_user_biz = pd.merge(review_user, business_data_selected, on = "business_id", how = "left")
    
    return review_user_biz



######################  Create the 3 samples

shuffle_indices = np.random.permutation(np.arange(len(reviews)))

###########  Reviews

reviews_selection = reviews[:sample_size]
reviews_selection = reviews_selection[ reviews_variables ]
sample_reviews = join_data( reviews_selection )
sample_reviews = create_yelp_columns( sample_reviews )



########### Business

groups = reviews[ ['business_id','review_id'] ].groupby('business_id').count()

print "Original number of business", len(groups)
print "Comments distribution:"
print "Std",np.std(groups)
print "Median",np.median(groups)
print "Mean",np.mean(groups)
groupfilter  =  groups
#pd.DataFrame(groupfilter.sort_values(by='review_id',ascending=False)).to_csv("yes.csv")
## percentile 0.98 has 200 reviews
## print len(groupfilter[groupfilter.review_id < 200])    #  ->  140922

groupfilter  = groupfilter[groupfilter.review_id < 200]

print "\nAfter looking at the distribution of reviews per business:\n"
print "We have a mean of ", np.mean(groupfilter), "comments per business"
sample_business_size = int(float(sample_size/np.mean(groupfilter)))
print "We want a sample of ", sample_size, "then we need ", sample_business_size

sample_biz_id = groupfilter.sample( sample_business_size )

businesses_selection = reviews[ reviews['business_id'].isin( sample_biz_id.index ) ].reset_index()
businesses_selection = businesses_selection[reviews_variables]
sample_business = join_data( businesses_selection )
sample_business = create_yelp_columns( sample_business  )



########### Users

groups = reviews[ ['user_id','review_id'] ].groupby('user_id').count()

print "Original number of people", len(groups)
print "Comments distribution:"
print "Std",np.std(groups)
print "Median",np.median(groups)
print "Mean",np.mean(groups)

personfilter =  groups
#pd.DataFrame(personfilter.sort_values(by='review_id',ascending=False)).to_csv("yes.csv")
## percentile 0.98 has 25 reviews
print len(personfilter[personfilter.review_id < 25])    #  ->  1007870

personfilter  = personfilter[personfilter.review_id < 25]

print "\nAfter looking at the distribution of reviews per person:\n"
print "We have a mean of ", np.mean(personfilter), "comments per person"
sample_user_size = int(float(sample_size/np.mean(personfilter)))
print "We want a sample of ", sample_size, "then we need ", sample_user_size

sample_usr_id = personfilter.sample( sample_user_size )

users_selection = reviews[ reviews['user_id'].isin( sample_usr_id.index ) ].reset_index()
users_selection = users_selection[reviews_variables]
sample_users = join_data( reviews_selection )
sample_users= create_yelp_columns( sample_users )


#####################################################################################################################
################################### 3. Functions to Clean Text and DataFrames  ######################################
#####################################################################################################################


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 
    #string = re.sub(r'\([^)]*\)', '', string)

    return string.strip().lower()


def clean_text_features(text_variable):

	clean_text = text_variable.map(lambda x: clean_str(x).split())
	clean_text = pd.Series(clean_text).map(lambda x: [word for word in x if word not in stopw] )

	binary_vectorizer = CountVectorizer(binary=True,ngram_range=(0,2), stop_words=stopw)
	clean_review = text_variable.map(lambda x: clean_str(x))
	binary_vectorizer.fit( clean_review )
	rev_text_binary = binary_vectorizer.transform( clean_review )

	return clean_text, rev_text_binary


def clean_sample_columns(sample_dataframe):

    remove_columns = ['business_id','user_id','review_id','usr_name','biz_name','text']
    dataframe_clean = sample_dataframe[ [i for i in sample_dataframe.columns  if i not in remove_columns ] ]
    dataframe_sparse = coo_matrix( dataframe_clean  )

    return dataframe_clean, dataframe_sparse



#####################################################################################################################
################################### 4. Create Files  ################################################################
#####################################################################################################################


def create_sample_files(sample_dataset, name_file):

    print("Cleaning samples..")

    text_output, text_binary_output = clean_text_features( sample_dataset['text'] )
    name_output, name_binary_output = clean_text_features( sample_dataset['biz_name'] )

    stack_binary_txt = hstack( [text_binary_output, name_binary_output ] )
    clean_output, sparse_output = clean_sample_columns( sample_dataset )
    sparse_binary_output = hstack( [stack_binary_txt, sparse_output] )

    pickle.dump( sparse_binary_output, open(output_path+name_file+'_sparse_binary.p', 'wb') )

    clean_output.loc[:,'biz_name'] = name_output
    clean_output.loc[:,'text'] = text_output

    pickle.dump( clean_output, open(output_path+name_file+'_clean.p', 'wb') )

    print (name_file+" = pickles' ready in: "+output_path)




########## Clean review text and name of the business

create_sample_files( sample_reviews, 'reviews')
create_sample_files( sample_business, 'reviews_biz')
create_sample_files( sample_users, 'reviews_usr')





###### 4. To Start LDA : example

#import vocabulary as v
#import lda
#voca = v.Vocabulary()
#docs = voca.read_corpus( rev_user_biz_clean['text'] )

