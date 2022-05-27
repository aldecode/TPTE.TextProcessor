from nltk import SnowballStemmer
from uk_stemmer import UkStemmer


def stem_ukr(tokens: [str]):
    stemmer = UkStemmer()
    return [stemmer.stemWord(token) for token in tokens]


def stem_eng(tokens: [str]):
    stemmer = SnowballStemmer(language='english')
    return [stemmer.stem(token) for token in tokens]
