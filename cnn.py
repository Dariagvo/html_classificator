from parser import Parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

import tensorflow as tf

from keras.layers import Add, Conv1D, ReLU, Dense, GlobalMaxPooling1D, Reshape, Embedding
from keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class MyCnn(keras.Model):
    def __init__(self, max_features, sequence_length):
        super().__init__()

        self.embedding_dim = 16
        self.sequence_length = sequence_length
        self.max_features = max_features

        self.skip_connection_lay = keras.Sequential(
            Conv1D(filters=32, kernel_size=1, padding='same'),
        )
        self.main_lay = keras.Sequential(layers=[
            Conv1D(filters=32, kernel_size=5, padding='same'),
            BatchNormalization(),
            ReLU(negative_slope=0.5),
            Conv1D(filters=32, kernel_size=5, padding='same'),
            BatchNormalization()
        ])
        self.finish_lay = keras.Sequential(layers=[
            ReLU(negative_slope=0.5),
            Conv1D(filters=1, kernel_size=5, padding='same'),
            Reshape(target_shape=(-1,)),
            ReLU(negative_slope=0.5),
            Dense(1, activation="sigmoid"),
        ])

    def call(self, inputs, training=None, mask=None):
        skip_data = self.skip_connection_lay(inputs)
        main_data = self.main_lay(inputs)
        add_data = Add()([skip_data, main_data])
        result = self.finish_lay(add_data)
        return result


def main():
    # качаем распаршенные страницы (очищенный вариант)
    table_path = "pars_data.csv"
    data = pd.read_csv(table_path, on_bad_lines='skip', sep=';', lineterminator='\n')
    data.dropna(axis=0, how='any', inplace=True)

    # почистить данные от мусора
    # pars = Parser()
    # data['text'] = data['text'].apply(pars.clean_text)
    # data.to_csv("pars_data.csv", index=False, sep=';')

    data.text = data.text.astype(str)

    # fit Tokenizer
    num_words = 1000000
    tokenizer = Tokenizer(num_words=num_words, oov_token="unk")
    tokenizer.fit_on_texts(data['text'].tolist())
    print('fit Tokenizer')

    # convert text to numb
    sequences = tokenizer.texts_to_sequences(data['text'].tolist())
    # ********************************
    # обрезаем данные от хэдеров и футеров
    # n = max(len(i) for i in sequences)
    # cut = (n - 7000) // 2
    # sequences = pad_sequences(sequences, maxlen=n-cut, truncating='post')
    # sequences = pad_sequences(sequences, maxlen=n, truncating='pre')
    # ********************************
    padded_sequences = pad_sequences(sequences, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['target'].tolist(), test_size=0.1, stratify=data['target'].tolist(), random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    print('convert text to numb')

    X_train = np.array(X_train).reshape(len(X_train), len(X_train[0]), 1).astype(np.float32)
    X_val = np.array(X_val).reshape(len(X_val), len(X_val[0]), 1).astype(np.float32)
    X_test = np.array(X_test).reshape(len(X_test), len(X_test[0]), 1).astype(np.float32)

    y_train = np.array(y_train).reshape((-1,1))
    y_val = np.array(y_val).reshape((-1,1))
    y_test = np.array(y_test).reshape((-1,1))

    # create cnn
    model = keras.Sequential()
    model.add(Embedding(num_words + 1, 16, input_length=len(X_train[0])))
    model.add(MyCnn(num_words, len(X_train[0])))

    model.compile(loss=keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.3), optimizer='Nadam', metrics=["accuracy", keras.metrics.AUC(curve='PR'),
                                                                                                                                keras.metrics.Precision(), keras.metrics.Recall()])
    print('create cnn')

    # Fit the model using the train and val datasets
    epochs = 10
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    print('Fit the model using the train and val datasets')

    plt.plot(history.history['loss'], label=' training data')
    plt.plot(history.history['val_loss'], label='validation data)')
    plt.title('Loss for Text Classification')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

    print(history.history)

    # plt.plot(history.history['CategoricalAccuracy'], label=' (training data)')
    # plt.plot(history.history['val_CategoricalAccuracy'],
    #          label='CategoricalCrossentropy (validation data)')
    # plt.title('CategoricalAccuracy for Text Classification')
    # plt.ylabel('CategoricalAccuracy value')
    # plt.xlabel('No. epoch')
    # plt.legend(loc="upper left")
    # plt.show()

    predictions = model.predict(X_test)
    print(predictions)
    result = pd.DataFrame(np.round(predictions), columns=['p'])
    result.to_csv("result.csv", index=False, sep=';')
    # print('acc:', accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
