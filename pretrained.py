import gensim.downloader
import numpy as np

#show all available models

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
print("Model Loaded")
print(glove_vectors.most_similar('camera'))
