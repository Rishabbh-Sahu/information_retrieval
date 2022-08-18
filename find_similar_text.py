# -*- coding: utf-8 -*-
"""
Created on tuesday 13-May-2021
@author: rishabbh-sahu
"""

import os
import pandas as pd
import time

from text_preprocessing import preprocessing,vectorizer

# configurations
FILE_PATH = os.path.join('data','articles.csv')
content_col = "text"
num_top_similar_docs = 1
ENCODING = "ISO-8859-1" #Can try utf-8 and others, depending on the raw text
# maximum terms to be considered for calculating documents similarity
max_features = 30000 #optimize this parameter based on your corpus

# Reading the csv file of texts/documents/queries (corpus)
df = pd.read_csv(FILE_PATH, delimiter=',', encoding=ENCODING)
print("Corpus is ready!!")

# Preprocess the corpus
data = [preprocessing.text_preprocessing(text) for text in df[content_col]]

# Learn vocabulary and inverse document frequency (idf), and get tf-idf matrix
text_vectorizer = vectorizer.TFIDF(max_features)
tfidf_mat = text_vectorizer.create_tfidf_features(data)
features = text_vectorizer.tfidf_vectorizor.get_feature_names()

print(f'Length of total features considered: {len(features)}')

# Letting user enter the text to find the closest 'sequnce of words' that matches this input 
print('Enter your text input here:')
search_text = input()
search_start = time.time()
sim_vecs, cosine_similarities = text_vectorizer.calculate_similarity(tfidf_mat, [search_text],top_k=num_top_similar_docs)
search_time = time.time() - search_start
print("search time: {:.2f} ms".format(search_time * 1000))
print("similar documents:")
text_vectorizer.show_similar_texts(data, cosine_similarities, sim_vecs)
