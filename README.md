# information_retrieval
Based on tf-idf matrix, searching for the closest documents within the list of documents. 

##### Getting started
create virtual environment<br>
install requirements

##### For building tf-idf matrix and getting similar documents - 
python find_similar_text.py

##### Where to use:
1) Whenever you need to browse thru huge number of articles to find the most closest one w.r.t. your task, you can use tf-idf based matrix multiplication to achieve this very quickly
2) For queries where your model fail to predict the right class (sentiments/intents etc.), this repo can help you identify queries which is of similar pattern or use same type of vocab
3) To merge similar type of patterns/documents since it's always better to have more discernible group

#### References - 
Data set is all about "Medium Articles", A collection of articles on ML, AI and data science. Download link - https://www.kaggle.com/hsankesara/medium-articles?select=articles.csv

