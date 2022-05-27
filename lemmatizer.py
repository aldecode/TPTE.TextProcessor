from nltk import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer


def lemmatize_ukr(tokens: [str]):
    lemmatizer = MorphAnalyzer()
    return [lemmatizer.parse(token)[0].normal_form for token in tokens]


def lemmatize_eng(tokens: [str]):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
