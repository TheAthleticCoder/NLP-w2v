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
word_vectors = model.wv

#print words similar to the word 'camera'
print(word_vectors.most_similar("camera"))

model.save("own_w2v.model")

def tsne_plot(model, word):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]
    #get close words
    close_words = model.wv.most_similar(word)
    #add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv.get_vector(word)]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv.get_vector(wrd_score[0])
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
    #run t-sne on array of vectors
    Y = tsne_model.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    #display scatter plot
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    # plt.xlim(x_coords.min()+0.005, x_coords.max()+0.005)
    # plt.ylim(y_coords.min()+0.005, y_coords.max()+0.005)
    return plt

words = ['comfortable','rating','crisp','best','work']

#plot for all words in a single plot
plt.figure(figsize=(24, 24))
for i,word in enumerate(words):
    # ax = plt.subplot(1, len(words), i+1)
    ax = plt.subplot(3,2, i+1)
    tsne_plot(model, word)
    plt.title(word)
plt.show()

