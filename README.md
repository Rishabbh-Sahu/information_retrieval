# Information Retrieval
An essential and key advantage of NLP is 'information retrieval' from the vast amount of information available (now a days), google search is the perfect example of this. Based on **tf-idf matrix (vectors), searching for the closest documents within the list of documents leveraging cosine similarity.**

#### How it works:
Based on the concept of matrix multiplication which is really fast, I thought of **representing every document/text/queries in the numeric form (else you can't perform matrix multiplication -> core idea of deep learning as well).** Tf-idf is one of the 'text vectorization' technique which is being used in order to transform corpus into an equivalent matrix (*tf-dif matrix a numeric representation*) form . Once you have the tf-idf vector (metrix) representation, you can leverage this metrix to compute cosine similarity with any other similar vector which can be used to find most closest doc/text/query present in the corpus. This enables us to find those closest/similar patterns in a very short duration. ***This method does not consider semantic/syntatic similarities between document/query/sentence hence it won't work when the objective is to retrieve document/query/sentence based on context.***  


#### Getting started:
create virtual environment<br>
install requirements

#### For building tf-idf matrix and getting similar documents: 
python find_similar_text.py <br>
(Above script supports user interactions, kindly check by entering text/pattern and the output would be the closest articles present based on your input)

#### Where to use:
1) Browsing through huge number of articles, to find the most closest one w.r.t. your task, you can use **tf-idf based matrix multiplication** to achieve it very quickly
2) For queries where your model fail to predict the right class (sentiments/intents etc.), this repo can help you identify queries which is of similar pattern or use same type of vocab
3) To merge similar type of patterns/documents since it's always better to have more discernible group

#### Further improvements:
1) By leveraging text pre-processing options like stemming, lemmitization. The advantage of using them upfront is to not only reduce the matrix sparsity but also help with more informed similarity matching using lemma/root of the word (which might get missed out due to inflectional endings)
2) Using different techniques of text vectorization
- Binary Term Frequency
- Bag of Words(BoW) Term Frequency
- (L1) Normalized Term Frequency
- (L2) Normalized tf-idf (currently in use)
- Word2Vec (kindly have a look at genesim for pre-trained embeddings) 
- flairNLP library (entials most of the available pre-learned embeddings)
3) Using semantic similarity from pre-trained language models (bert, albert , transformer based model etc). Using BERT implemented here - https://github.com/Rishabbh-Sahu/semantic_lookalike_transformers

#### References: 
Data set is all about "Medium Articles", A collection of articles on ML, AI and data science. Download link - https://www.kaggle.com/hsankesara/medium-articles?select=articles.csv

