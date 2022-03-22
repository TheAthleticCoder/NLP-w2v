import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import random

#open txt file
data_file = open('derived_data.txt', 'r')
total_sentences = data_file.read().splitlines() #split lines
data_file.close()

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