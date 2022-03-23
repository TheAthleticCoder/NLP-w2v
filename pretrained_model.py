import gensim.downloader
import numpy as np

#we call a pretrained library from the gensim w2v library 
#and use that for our analysis and report

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
print("Model Loaded")
print(glove_vectors.most_similar('camera'))
