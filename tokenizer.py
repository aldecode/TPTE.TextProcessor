import string

from nltk import word_tokenize
from nltk.corpus import stopwords


def tokenize_ukr(text: str):
    with open('stopwords_ua.txt') as sw:
        stopwords_list = sw.readline().split(' ') + list(string.punctuation) + list(range(10))
        return [token for token in word_tokenize(text) if not token.lower() in stopwords_list]


def tokenize_eng(text: str):
    stopwords_list = list(stopwords.words('english')) + list(string.punctuation) + list(range(10))
    return [token for token in word_tokenize(text) if not token.lower() in stopwords_list]
