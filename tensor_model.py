import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

#open txt file
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
# print(len(total_sentences))
total_sentences = total_sentences[0:1500] #remember to change UNK counter
# print(total_sentences)
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
    if temp_word_counts[word] >= 2: 
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
vocab_size = len(vocab)
words_in_sentence = []
for sentence in total_sentences:
    words_in_sentence.append(sentence.split())

word2int = {}
int2word = {}
for i, word in enumerate(vocab):
    word2int[word] = i
    int2word[i] = word

WINDOW_SIZE = 1
data = []
for sentence in words_in_sentence:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word])

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

print(x_train.shape, y_train.shape)

EMBEDDING_DIM = 50 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
# prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
prediction = tf.nn.sigmoid(tf.matmul(hidden_representation, W2) + b2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!

#update only a few weights
#implement negative sampling through loss function
# def get_loss(predictions, labels):
#     return -tf.reduce_sum(labels*tf.log(predictions))

def cosine_dist(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=prediction))
with tf.control_dependencies(update_ops):
   optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

n_iters = 500
i = 0
for _ in range(n_iters):
    sess.run(optimizer, feed_dict={x: x_train, y_label: y_train})
    print(i)
    print('loss is : ', sess.run(loss, feed_dict={x: x_train, y_label: y_train}))
    i += 1

vectors = sess.run(W1 + b1)


def closest_words(word_index, vectors):
    results = []
    word = int2word[word_index]
    query_vector = vectors[word_index]
    for index in range(10):
        min_dist = 10000 # to act like positive infinity
        min_index = -1
        for i in range(len(vectors)):
            if cosine_dist(vectors[i], query_vector) < min_dist and not np.array_equal(vectors[i], query_vector):
                min_dist = cosine_dist(vectors[i], query_vector)
                min_index = i
        closest_word = int2word[min_index]
        query_vector = vectors[min_index]
        results.append((closest_word,min_dist))
    return results

# #top 10 closest words to 'king'
# word_eval = ['update']
word_eval = ['comfortable','rating','crisp','best','work']
for word in word_eval:
    word_index = word2int[word]
    print(word, ':', closest_words(word_index, vectors))

#plot tsne for above words
def plot_closest_words(word_index, vectors):
    close_words = closest_words(word_index, vectors)
    x_data = []
    y_data = []
    for word, dist in close_words:
        x_data.append(vectors[word_index])
        y_data.append(vectors[word])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_data)
    y_tsne = tsne.fit_transform(y_data)
    plt.scatter(x_tsne[:,0], x_tsne[:,1], c='r')
    plt.scatter(y_tsne[:,0], y_tsne[:,1], c='b')
    plt.show()

#plot tsne graph for 'comfortable','rating','crisp','best','work'
word_eval = ['comfortable','rating','crisp','best','work']
for word in word_eval:
    word_index = word2int[word]
    plot_closest_words(word_index, vectors)