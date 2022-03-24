import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

#read csv file
input_data = pd.read_csv('U_reduced.csv', index_col=0)

model_dict = {}
for i in list(input_data.index):
    model_dict[i] = list(input_data.loc[i])


#display top 10 closest words to 'camera'
def top_similar(word, model_dict, n=10):
    word_vec = model_dict[word]
    similarities = []
    for w in model_dict:
        if w != word:
            similarities.append((w, np.dot(word_vec, model_dict[w])))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:n]

topit = top_similar('camera', model_dict)
print(topit)

#cosine similarity to find closest words
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_similar_words(words, model_dict, n=10):
    total_word_list = []
    for word in words:
        similar_list = top_similar(word, model_dict, n)
        for w in similar_list:
            total_word_list.append(w[0])
    total_word_vectors = np.array([model_dict[w] for w in total_word_list])
    return total_word_list, total_word_vectors

def tsne_plot(word, model_dict, n):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]
    close_words = top_similar(word, model_dict, n)

    arr = np.append(arr, np.array([model_dict[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model_dict[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
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

plt.figure(figsize=(24, 24))
for i,word in enumerate(words):
    # ax = plt.subplot(1, len(words), i+1)
    ax = plt.subplot(3,2, i+1)
    tsne_plot(word, model_dict, n=10)
    plt.title(word)
plt.show()
