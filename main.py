import string
from nltk import word_tokenize, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from uk_stemmer import UkStemmer

def analyze_text_ua(txt):
    analyzer = MorphAnalyzer()
    stemmer = UkStemmer()

    with open('stopwords_ua.txt') as sw:
        stopwords = sw.readline().split(' ') + list(string.punctuation) + list(range(10))

    tokens = [token for token in word_tokenize(txt) if not token.lower() in stopwords]
    lemms = [analyzer.parse(token)[0].normal_form for token in tokens]
    stems = [stemmer.stemWord(token) for token in tokens]

    print("Stopwords: {}".format(stopwords))

    return tokens, lemms, stems


def analyze_text_en(txt):
    stemmer = SnowballStemmer(language='english')
    lemmatizer = WordNetLemmatizer()
    stop_words = list(stopwords.words('english')) + list(string.punctuation) + list(range(10))
    tokens = [token for token in word_tokenize(txt) if not token.lower() in stop_words]
    lemms = [lemmatizer.lemmatize(token) for token in tokens]
    stems = [stemmer.stem(token) for token in tokens]

    print("Stopwords: {}".format(stop_words))

    return tokens, lemms, stems


if __name__ == '__main__':
    print("\n<Ukrainian Text>")
    with open('text_ua.txt') as textio:
        text = ''.join(textio.readlines())
        tokens, lemms, stems = analyze_text_ua(text)
        print("Tokens: {}".format(tokens))
        print("Lemms: {}".format(lemms))
        print("Stems: {}".format(stems))

    print("\n<English Text>")
    with open('text_en.txt') as textio:
        text = ''.join(textio.readlines())
        tokens, lemms, stems = analyze_text_en(text)
        print("Tokens: {}".format(tokens))
        print("Lemms: {}".format(lemms))
        print("Stems: {}".format(stems))
