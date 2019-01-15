import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from .sequence_classifier import SequenceClassifier


class NeuralNetworksClassifier(SequenceClassifier):
    def __init__(self, name='Neural Networks', transformer=None, memory_units=100, max_seq_length=500, activation='sigmoid',
                 loss_function='categorical_crossentropy', optimizer='adam', metrics=None, epochs=3, batch_size=64, verbose=1):
        super(NeuralNetworksClassifier, self).__init__(name)
        self.max_seq_length = max_seq_length
        self.memory_units = memory_units
        self.activation = activation
        self.loss_function = loss_function
        self.optimizer = optimizer
        if metrics is None:
            self.metrics = ['accuracy']
        else:
            self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        X = self.transform(X)
        y = to_categorical(y)
        self.model_ = Sequential()
        self.model_.add(LSTM(self.memory_units, input_shape=(self.max_seq_length, 1)))
        self.model_.add(Dense(len(y[0]), activation=self.activation))
        self.model_.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def _predict(self, X):
        X = self.transform(X)
        return np.argmax(self.model_.predict(X), axis=1)

    def transform(self, X):
        X = sequence.pad_sequences(X, maxlen=self.max_seq_length)
        X = X[:, :, np.newaxis]
        return X

