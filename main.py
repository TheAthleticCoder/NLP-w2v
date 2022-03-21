import numpy as np
import pandas as pd
from numpy import array
from scipy.linalg import svd
from numpy import diag
from numpy import dot

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
        
#create vocabulary list    
vocab=list(set((" ".join(total_sentences)).split()))
# print(len(vocab))
# print(len(word_counts))

#find different words in the vocabulary
# diff_words = []
# for word in vocab:
#     if word not in word_counts:
#         diff_words.append(word)


#2d sentences with words
words_in_sentence = []
for sentence in total_sentences:
    words_in_sentence.append(sentence.split())

# def MinWords(words_in_sentence):
#     #find the minimum number of words in a sentence
#     min_words = min(len(sentence) for sentence in words_in_sentence)
#     return min_words

# print(MinWords(words_in_sentence))

# #initialize window_size 
window_size = 1

w=[]
for i in total_sentences:
    i=i.split(' ') #if i is list
    for k in range(len(i)-window_size+1):
        for l in range(k+1,k+window_size+1):
            if l<=len(i)-1:
                w.append([i[k],i[l]])
                #append the reverse as well
                w.append([i[l],i[k]])

# w1=[x[::-1] for x in w]
# w.extend(w1)
# print(len(w))

#create cooccurence matrix
a = np.zeros((len(vocab),len(vocab)))
cooccurence_matrix=pd.DataFrame(a,index=vocab,columns=vocab)
count = 0
for i in w:
    #skip empty elements
    if i[0]=='' or i[1]=='':
        continue
    cooccurence_matrix.at[i[0],i[1]] +=1
    count+=1
    if count%10000==0:
        print(count)

#the csv file generation takes quite some time, so relax and let the code run
def csv_gen(cooccurence_matrix):
    cooccurence_matrix.to_csv('cooccurence_matrix.csv',sep=',',header=True,index=True)


csv_gen(cooccurence_matrix)
A = array(cooccurence_matrix)
print("D")
U, s, V = svd(A)
print("O")
S = diag(s)
print("K")

#reduce dimensionality
k = 100
U_reduced = U[:,:k]
S_reduced = S[:k,:k]
V_reduced = V[:k,:]

#print all their shapes
print(U_reduced.shape)
print(S_reduced.shape)
print(V_reduced.shape)

U_reduced_df = pd.DataFrame(U_reduced, index=vocab)
print(U_reduced_df.head(10))
U_reduced_df.to_csv('U_reduced.csv',sep=',',header=True,index=True)
    






