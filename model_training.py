import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

# ------------------------------------PART 3: preprocess the data for ML ---------------------------------------------#
# main source: Nagy, Peter: LSTM Sentiment Analysis | Keras,
# https://github.com/nagypeterjob/Sentiment-Analysis-NLTK-ML-LSTM/blob/master/lstm.ipynb


def data_preprocessing():
    """Reads the data, processes them and creates the feature matrix."""
    data = pd.read_csv('new_data.csv')
    # we only keep the cleaned comments and the labels
    data = data[["Clean", "Label"]]
    # max the number of features
    # DOCs https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
    # the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data["Clean"].values)
    X = tokenizer.texts_to_sequences(data['Clean'].values)
    X = pad_sequences(X)  # makes all sequences the same length
    # print(X.shape) (1296, 253)
    return max_features, X, data

# ---------------------------------------- PART 4: create the RNN ------------------------------------------------#


def lstm_network(feature_nr, feature_matrix):
    """Defines what model to use and creates the neural network."""
# creating the LSTM network
    embed_dim = 128
    lstm_out = 196
    model = Sequential()
    # input: vocab size, i.e. unique words, embed dim, words will be embedded in vectors of 128 dim
    model.add(Embedding(input_dim=feature_nr, output_dim=embed_dim, input_length=feature_matrix.shape[1]))
    model.add(SpatialDropout1D(0.4))  # dropout, to avoid overfitting
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))  # softmax activation function
    # working with an unbalanced dataset: accuracy is not the best metric
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# -------------------------------------------PART 5: train the RNN and evaluate performance----------------------------#
def train_lstm(model, feature_matrix, data):
    """Performs train-test-split and trains the model. Also creates validation set and evaluates the model."""
# train and test dataset
    y = pd.get_dummies(data["Label"]).values  # it avoids a value error
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=42)
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Test data: {X_test.shape}, {y_test.shape}")
    # validation set
    validation_size = 80
    X_validate = X_test[-validation_size:]
    y_validate = y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    y_test = y_test[:-validation_size]
    # training the network
    # reducing the effects of unbalanced dataset
    class_weight = {0: 1.,
                    1: 150.}
    batch_size = 32
    model.fit(X_train, y_train, epochs=7, batch_size=batch_size, class_weight=class_weight, verbose=2)

    # Evaluate the model on the test data
    results = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
    print(f"Loss and Accuracy: {results}")

    return model, X_test, y_test, X_validate, y_validate

# -----------------------------------------PART 6: use the model to make predictions-----------------------------------#


def predict(model, X_validate, X_test, y_validate):
    """Performs predictions and prints accuracy"""
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):
        print(X_validate[x].shape)
        result = model.predict((X_validate[x]).reshape(1, X_test.shape[1]), batch_size=1, verbose=1)[0]
        if np.argmax(result) == np.argmax(y_validate[x]):
            if np.argmax(y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print(f"Positive accuracy: {round(pos_correct / pos_cnt * 100, 3)}%")
    print(f"Negative accuracy: {round(neg_correct / neg_cnt * 100, 3)}%")
    # results: Positive accuracy: 55.556%
    # Negative accuracy: 95.775%


def sentiment_analysis():
    """Calls all other functions to perform sentiment analysis on the dataset."""
    max_features, X, data = data_preprocessing()
    sentiment_analyser = lstm_network(max_features, X)
    model, X_test, y_test, X_validate, y_validate = train_lstm(sentiment_analyser, X, data)
    predict(model, X_validate, X_test, y_validate)


sentiment_analysis()



### Bonus function to classify own comments:
#def analyse_own_text(model, comment):
#    own_comment = [comment]
#    tokenizer = Tokenizer()
#    data = tokenizer.texts_to_sequences(own_comment)
#    data = pad_sequences(data, maxlen=253, dtype='int32', value=0)
#    sentiment = model.predict(data, verbose=2)
#    if np.argmax(sentiment) == 0:
#        print("negative")
#    elif np.argmax(sentiment) == 1:
#        print("positive")
