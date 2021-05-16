# information_retrieval
Based on tf-idf matrix, searching for the closest documents within the list of documents.

#### How it works
Based on the concept of matrix multiplication which is really fast, I thought of representing every document/text/queries in the numeric form (else you can't perform matrix multiplication -> core idea of d). Tf-idf is one of the text vectorization method which i've used here in order to create numeric representation of the input corpus. Once you have the tf-idf vector representation, you can leverage this metrix to compute cosine similarity with any other similar vector. 

#### Getting started
create virtual environment<br>
install requirements

#### For building tf-idf matrix and getting similar documents - 
python find_similar_text.py

#### Where to use:
1) Whenever you need to browse thru huge number of articles to find the most closest one w.r.t. your task, you can use tf-idf based matrix multiplication to achieve this very quickly
2) For queries where your model fail to predict the right class (sentiments/intents etc.), this repo can help you identify queries which is of similar pattern or use same type of vocab
3) To merge similar type of patterns/documents since it's always better to have more discernible group

#### References - 
Data set is all about "Medium Articles", A collection of articles on ML, AI and data science. Download link - https://www.kaggle.com/hsankesara/medium-articles?select=articles.csv

