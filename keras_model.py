from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import numpy as np

data_file = open('derived_data.txt', 'r')
total_sentences = data_file.read().splitlines() #split lines
data_file.close()

#choose the sentences if it contains atleast one of the words in the list
def choose_sentences(sentences, words):
    chosen_sentences = []
    for sentence in sentences:
        if any(word in sentence.split() for word in words):
            chosen_sentences.append(sentence)
    return chosen_sentences

total_sentences = choose_sentences(total_sentences, ['camera','comfortable','rating','crisp','best','work'])

total_sentences = total_sentences[0:5000] #remember to change UNK counter
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(total_sentences)
word2id = tokenizer.word_index

# build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in total_sentences]

vocab_size = len(word2id)
embed_size = 100
window_size = 1 # context window size

# print('Vocabulary Size:', vocab_size)
# print('Vocabulary Sample:', list(word2id.items())[10:40])

def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word   = []            
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([words[i] 
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])
            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)
            
            
# Test this out for some samples
# i = 0
# for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
#     if 0 not in x[0]:
#         print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
    
#         if i == 10:
#             break
#         i += 1

import keras.backend as K
from keras.models import Sequential
from tensorflow import keras
from keras.layers import Dense, Embedding, Lambda
from keras.utils import np_utils

# build CBOW architecture
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# view model summary
print(cbow.summary())

# visualize model structure
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False, 
                rankdir='TB').create(prog='dot', format='svg'))

for epoch in range(1, 4):
    loss = 0.
    i = 0
    for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100 == 0:
            print('Processed {} (context, word) pairs'.format(i))

    print('Epoch:', epoch, '\tLoss:', loss)
    print()


weights = cbow.get_weights()[0]
weights = weights[1:]
print(weights.shape)
import pandas as pd
df = pd.DataFrame(weights, index=list(id2word.values())[1:])
#save it
df.to_csv('embedding_weights.csv')
df.head()

from sklearn.metrics.pairwise import euclidean_distances

# compute pairwise distance matrix
distance_matrix = euclidean_distances(weights)
print(distance_matrix.shape)

# view contextually similar words
similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:11]+1] 
                   for search_term in ['camera','comfortable','rating','crisp','best','work']}

print(similar_words)