import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import random

#open txt file
data_file = open('derived_data.txt', 'r')
total_sentences = data_file.read().splitlines() #split lines
data_file.close()

#create a dictionary of all vocabulary words and their counts
temp_word_counts = {}
for sentence in total_sentences:
    for word in sentence.split():
        if word not in temp_word_counts:
            temp_word_counts[word] = 1
        else:
            temp_word_counts[word] += 1
    
#for word_counts less than 3, remove the word and its count and add 'UNK' and its count
word_counts = {}
unk_counter = 0
for word in temp_word_counts:
    if temp_word_counts[word] >= 3:
        word_counts[word] = temp_word_counts[word]
    else:
        unk_counter += temp_word_counts[word]
word_counts['<UNK>'] = unk_counter

#change sentences based on the presence of <UNK> words
temp = []
for sentence in total_sentences:
    for word in sentence.split():
        if word not in word_counts:
            sentence = sentence.replace(word, '<UNK>')        
    temp.append(sentence)
total_sentences = temp

vocab=list(set((" ".join(total_sentences)).split()))
words_in_sentence = []
for sentence in total_sentences:
    words_in_sentence.append(sentence.split())

model = Word2Vec(words_in_sentence, min_count=5, sg=0, hs=0, negative=5, vector_size=100, window=2, epochs=10)
# vocabulary = word2vec.wv.vocab
# print(vocabulary)
word_vectors = model.wv
print(word_vectors.most_similar("camera"))
# print(words_in_sentence)
# print(random.choice(model.wv.index_to_key))

model.save("word2vec.model")

# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []

#     new_values = tsne_model.fit_transform(model.wv.get_vector_representation())

#     for word in list(model.wv.index_to_key):
#         tokens.append(model.wv.get_index(word))
#         labels.append(word)
    
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()

# tsne_plot(model)