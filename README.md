# **NLP-w2v**

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

-----
## ***Objectives:***
This repository intends to familiarise you with the above techniques by implementing one of the **frequency-based modelling approaches** and comparing it to embeddings acquired using one of the word2vec variations. You'll start by attempting to obtain embeddings using the **Singular Value Decomposition (SVD)** method with a corpora given. The **CBOW implementation** of word2vec with **Negative Sampling** would then be used. Following that, a brief analysis would be conducted, showing the differences in the quality of the obtained embeddings.

-----
## ***File Structure:***
1. `preprocess.py` contains the code used for preprocessing the text and separating them based on **spaces**.
2. `read_json.py` contains the code to read limited number of sentences from the large dataset(about 500MB) and store them after doing the necessities in `derived_data.txt`.
3. `svd.py` implements the co-occurence matrix and SVD. It generates `cooccurence_matrix.csv` and `U_reduced.csv` which are important components of the concept. 
4. `model1.py` Implements the first model given in the document. It displays the most similar words for the word 'camera' and generates **t-SNE graphs** for 5 different grammatical words.
5. `model2.py` Implements the second model given in the document. It displays the most similar words for the word 'camera' and generates **t-SNE graphs** for 5 different grammatical words.
6. `pretrained_model.py` runs the code on a pretrained word2vec model. It is used to compare with the 2 created models above. 
7. `own_w2v.model` contains loadable model from `model1.py`
8. `word2vec.model` contains loadable model from `model2.spy`

-----

## ***Execution:***

The code can be fully executed in the following manner:
```py
python3 <filename>.py
```
The order in which the files are run is:

1. `read_json.py`  
2. `svd.py`     
3. `model1.py`  
4. `model2.py`  
5. `pretrained_model.py`

-----






