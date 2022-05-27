import bag_of_words
import lemmatizer
import oh_encoder
import stemmer
import tokenizer


if __name__ == '__main__':
    print("\n<Ukrainian Text>")
    with open('text_ua.txt') as textio:
        ukr_text = ''.join(textio)

    ukr_tokens = tokenizer.tokenize_ukr(ukr_text)
    print("Tokens: {}".format(ukr_tokens))

    ukr_lemms = lemmatizer.lemmatize_ukr(ukr_tokens)
    print("Lemms:  {}".format(ukr_lemms))

    ukr_stems = stemmer.stem_ukr(ukr_tokens)
    print("Stemms: {}".format(ukr_stems))

    # Note: IT takes top 10 most frequent words and building BoW for each sentence.
    ukr_bag = bag_of_words.get_bag_of_words(ukr_text)
    print("Bag of Words for top frequent words per sentence:\n {}".format(ukr_bag))

    # To many lems in text so, I will take only top 10
    ukr_encoding = oh_encoder.onehot_encode(ukr_tokens[:10])
    print("One Hot Encoding:\n{}".format(ukr_encoding))

    print("\n<English Text>")
    with open('text_en.txt') as textio:
        eng_text = ''.join(textio)

    eng_tokens = tokenizer.tokenize_eng(eng_text)
    print("Tokens: {}".format(eng_tokens))

    eng_lemms = lemmatizer.lemmatize_eng(eng_tokens)
    print("Lemms:  {}".format(eng_lemms))

    eng_stems = stemmer.stem_eng(eng_tokens)
    print("Stemms: {}".format(eng_stems))

    # Note: IT takes top 10 most frequent words and building BoW for each sentence.
    eng_bag = bag_of_words.get_bag_of_words(eng_text)
    print("Bag of Words for top frequent words per sentence:\n {}".format(eng_bag))

    # To many lems in text so, I will take only top 10.
    eng_encoding = oh_encoder.onehot_encode(eng_tokens[:10])
    print("One Hot Encoding:\n{}".format(eng_encoding))
