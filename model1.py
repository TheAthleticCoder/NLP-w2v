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

# words = ['old','little','work','wife','negative']
words = ['big']
# word_vectors = np.array([model_dict[w] for w in words])
# print(word_vectors)

#display top 10 closest words to 'camera'
def top_similar(word, model_dict, n=10):
    word_vec = model_dict[word]
    similarities = []
    for w in model_dict:
        if w != word:
            similarities.append((w, np.dot(word_vec, model_dict[w])))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:n]

# topit = top_similar('camera', model_dict)
# print(topit)

# topit = top_similar('camera', model_dict)
# print(topit)

#cosine similaroty to find closest words
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

words, word_vectors = get_similar_words(words, model_dict, n=10)
print("words:", words)

#Display the top-10 word vectors for five different words using t-SNE (or such methods) on a 2D plot.
def tsne_plot(word_vectors, words):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(word_vectors)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    #plot using sns
    sns.scatterplot(x, y)
    # for i, word in enumerate(words):
    #     plt.scatter(x[i], y[i])
    #     plt.annotate(word, xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()




tsne_plot(word_vectors, words)


