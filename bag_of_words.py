import heapq
import re

import numpy as np
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def get_bag_of_words(text: str):
    sentences = sent_tokenize(text)
    wordfreq = {}
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        sentences[i] = re.sub(r'\W', ' ', sentences[i])
        sentences[i] = re.sub(r'\s+', ' ', sentences[i])

    # Creating the Bag of Words model
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            if word not in wordfreq.keys():
                wordfreq[word] = 1
            else:
                wordfreq[word] += 1
    wordfreq = heapq.nlargest(10, wordfreq, key=wordfreq.get)
    print("Top 10 frequent words: {}".format(wordfreq))
    # Building the Bag of Words model
    bow = []
    for sentence in sentences:
        vector = []
        for word in wordfreq:
            if word in word_tokenize(sentence):
                vector.append(1)
            else:
                vector.append(0)
        bow.append(vector)
    return np.asarray(bow)
