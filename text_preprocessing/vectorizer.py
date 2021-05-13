# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF:

    def __init__(self,max_features=30000):
        '''
        model_layer: Model layer to be used to create the tokenizer for
        max_seq_length: int - maximum number of tokens to keep in a sequence
        '''
        super(TFIDF,self).__init__()
        self.max_features = max_features
        self.tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                                stop_words=None, ngram_range=(1, 1), max_features=self.max_features,
                                                norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)

    def create_tfidf_features(self,corpus):
        '''
        Creates a tf-idf matrix for the input corpus of data
        param corpus: input text/documents collection
        return: tfidf matrix
        '''
        #     tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
        #                                        stop_words='english', ngram_range=(1, 1), max_features=max_features,
        #                                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        #                                        max_df=max_df, min_df=min_df)
        tfidf_mat = self.tfidf_vectorizor.fit_transform(corpus)
        print('tfidf matrix successfully created.')
        return tfidf_mat

    def calculate_similarity(self, tfidf_mat, text_str, top_k=10):
        """ Vectorizes the text/document via `vectorizor` and calculates the cosine similarity of
        the text/document and tfidf_mat (all the documents) and returns the `top_k` similar documents."""

        # Vectorize the query to the same length as documents
        text_vec = self.tfidf_vectorizor.transform(text_str)
        # Compute the cosine similarity between text_vec and all the texts/documents
        cosine_similarities = cosine_similarity(tfidf_mat, text_vec).flatten()
        # Sort the similar documents from the most similar to less similar and return the indices
        most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k - 1:-1]
        return (most_similar_doc_indices, cosine_similarities)

    def show_similar_texts(self,df, cosine_similarities, similar_doc_indices):
        """ Prints the most similar documents using indices in the `similar_doc_indices` vector."""
        counter = 1
        for index in similar_doc_indices:
            print('Top-{}, Similarity = {}'.format(counter, cosine_similarities[index]))
            print('body: {}, '.format(df[index]))
            print()
            counter += 1