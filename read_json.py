import pandas as pd
import numpy as np
from preprocess import preprocess
import json

#We have called the preprocess function to get the dataframe
sentence_list = []
#only open first 10 sentences in the file
with open('Electronics_5.json', 'r') as f:
    for i, line in enumerate(f):
        if i < 5000:
            #print type of line
            sentence_list.append(preprocess(json.loads(line)['reviewText']))
        else:
            break

total_sentences = []
for line in sentence_list:
    sent = line.replace('\n', ' ').split(" . ")
    for l in sent:
        total_sentences.append(l)
    
total_sentences = list(filter(None, total_sentences))

#remove all sentences with 1 letter or only 1 word
total_sentences = [x for x in total_sentences if len(x.split()) > 1]

#store all sentences in 'derived_data.txt'
with open('derived_data.txt', 'w') as filehandle:
    for listitem in total_sentences:
        filehandle.write('%s\n' % listitem)


