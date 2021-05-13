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
content_col = 'text'

# Reading the csv file of texts/documents/queries (corpus)
df = pd.read_csv(FILE_PATH, delimiter=',', encoding='ISO-8859-1')
# df = pd.read_csv(FILE_PATH, delimiter=',', encoding='utf-8')
print('Corpus is ready!!')

# Preprocess the corpus
data = [preprocessing.text_preprocessing(text) for text in df[content_col]]

# Learn vocabulary and inverse document frequency (idf), and get tf-idf matrix
text_vectorizer = vectorizer.TFIDF(max_features=20000)
tfidf_mat = text_vectorizer.create_tfidf_features(data)
features = text_vectorizer.tfidf_vectorizor.get_feature_names()

print(f'Length of total features considered: {len(features)}')

user_question = ['Our hopes were sky high. Bright-eyed and bushy-tailed, the industry was ripe for a new era of ']
search_start = time.time()
sim_vecs, cosine_similarities = text_vectorizer.calculate_similarity(tfidf_mat, user_question,top_k=1)
search_time = time.time() - search_start
print("search time: {:.2f} ms".format(search_time * 1000))
print(f'similar documents:')
text_vectorizer.show_similar_texts(data, cosine_similarities, sim_vecs)
