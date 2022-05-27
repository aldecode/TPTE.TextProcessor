from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def onehot_encode(tokens: [str]):
    label_encoder = LabelEncoder()
    encoded_integer = label_encoder.fit_transform(tokens)
    encoded_integer = encoded_integer.reshape(len(encoded_integer), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    return onehot_encoder.fit_transform(encoded_integer)
