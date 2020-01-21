#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
import numpy as np
import re
import pandas as pd
import string
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) 


# Without tokenizing the comments I am not sure how I can stem them quickly.
# I could just go over each word and stem as I go. Seems like it would take awhile.

# In[4]:


train_comments = pd.read_csv("training_copy.dat", sep="\n", header=None)
test_comments = pd.read_csv("test_copy.dat", sep="\n", header=None)

#Made all letters lowercase to remove any matching issues
train_comments_lower = train_comments[0].str.lower()
test_comments_lower = test_comments[0].str.lower()

#Converted the lines of training and test data to list and then combined them into a single corpus
train_comm_list = train_comments_lower.tolist()
test_comm_list = test_comments_lower.tolist()
combined_comments = train_comm_list + test_comm_list


# In[25]:


len(train_comm_list)


# In[26]:


len(test_comm_list)


# TfidfVectorizer is a powerful utility from scikit-learn. It creates a 2d array of features from documents. The matrix contains numerical data that reflects the importance of words relative to the document while reducing the importance of words that appear frequently in every document.
# 
# TF - Term frequency. Longer documents will have more occurences of certain words and so normalization via: 
# 
# TF(t) = term frequency/number of total terms in document
# 
# IDF - Certain terms, like stop words, will appear multiple times over many documents, and so their importance needs to be scaled down, while other words are more rare and their importance needs to be scaled up.
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

# In[24]:


def tf_idf_tokenizer(text):
    '''Custom tokenizer for  sklearn's tf-idf vectorizer. Strips away punctuation, digits, and 
    removes common word endings, such as ing or ed, to get a base word.
    Note: tf-idf vectorizer does offer the utility to remove stop words, make all letters lower case,
    and a few other utilities. I did not use these features at the time when I created this project,
    but I have since increased my familiarity with them.'''
    tokenized_words = [word for word in nltk.word_tokenize(text) if len(word)>1]
    stripped_punc_words = [''.join(char for char in word if char not in string.punctuation) for word in tokenized_words]
    stripped_num_words = [''.join(char for char in word if char not in string.digits) for word in stripped_punc_words]
    stemmed_words = [stemmer.stem(words) for words in stripped_num_words]
    trimmed_stops = []
    for word in stemmed_words:        
        if word not in stop_words:
            trimmed_stops.append(word)
    return trimmed_stops


# In[27]:


'''I chose to have the tf-idf vectorizer return 50 features and have them be between 1 and two words, partly due to run time.'''
tfidf=TfidfVectorizer(use_idf=True,analyzer='word',tokenizer=tf_idf_tokenizer,max_features=50,ngram_range=(1,2))


# In[28]:


# Returns the term document matrix
combo_tfidf_vectors = tfidf.fit_transform(combined_comments)


# In[29]:


# Used this just to get a visual of what sort of numbers I had in the array.
# They are more relative really,
# combo_tfidf_vectors.todense()


# In[30]:


# I wanted to ensure that all of the training and test documents were included
combo_tfidf_vectors.shape


# In[31]:


# Returns the most important terms from the given documents
# tfidf.vocabulary_


# In[32]:


training_tfidf = combo_tfidf_vectors[:18506,:50]


# In[33]:


test_tfidf = combo_tfidf_vectors[18506:36947,:250]


# In[34]:


type(test_tfidf)


# row=doc,col=word

# In[35]:


'''I like to look at tf-idf vectorizer and cosine_similarity as the linchpins of this program. 
The cosine similarity of course returns the similarity between every document in the training data 
to that found in the test data. It is from this matrix that I was able to perform K Nearest Neighbors
on.'''
distances = cosine_similarity(training_tfidf,test_tfidf)


# In[38]:


# Verifying counts for training and test docs
print(len(distances)) # # of rows = test docs
print(len(distances[0]))# # of columns = training docs


# In[39]:


distances


# In[40]:


'''Very helpful numpy function! It returns an array that is the same size 
as the one passed in. Each row of the returned array holds the indices of 
the training documents that are farthest to closest relative to that row
(test document).'''
sorted_distances = distances.argsort()


# In[41]:


sorted_distances


# In[42]:


sorted_distances[18440,-5:] #returns the top 5 training docs similar to the test doc


# I have a KNN function that takes in x = index of test doc & y = k-nearest neighbors of training documents. From there I get their respective ratings. I determine which rating/class label has the highest representation, +1 or -1, and choose that rating to add to a list of ratings.

# In[43]:


distances[0,3] #Confirmed that these are these highest similarities. The incoming 


# In[44]:


# train_comments = train_comments.str.split("\t",n=1,expand =  True)
comments = pd.read_csv("training_copy.dat", sep="\n", header=None)
comments = comments[0].str.lower()
comments = comments.str.split("\t",n=1,expand =  True)
ratings = comments[0]


# In[45]:


ratings[1]


# In[46]:


def most_freq(inc_list):
    return max(set(inc_list), key = inc_list.count) 

def knn(test_index,k):
    nearest_neighbors = sorted_distances[test_index,-abs(k):] #get the indices of the k nearest neighbors
    neighbor_ratings = [] 
    for neighbor in nearest_neighbors: 
        neighbor_ratings.append(ratings[neighbor])
    rating = most_freq(neighbor_ratings)
    return rating


# In[47]:


def create_list_knn(k):
    predicted_test_ratings = []
    for i in range(len(train_comments)):
        predicted_test_ratings.append(knn(i,k))
    f= open("joabrb22_format.dat","w+")
    for rating in predicted_test_ratings:
        f.write(rating+"\n")


# In[48]:


create_list_knn(7)

