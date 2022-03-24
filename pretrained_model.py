import gensim.downloader
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

#we call a pretrained library from the gensim w2v library 
#and use that for our analysis and report

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
print("Model Loaded")

words = ['camera','comfortable','rating','crisp','best','work']

for i in range(len(words)):
    print("Word: ",words[i])
    print("Similar Words: ",glove_vectors.most_similar(i))
    print("\n")



