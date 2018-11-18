from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sequence_classifier import SequenceClassifier
from sequence_transformer import SequenceTransformer


class NeuralNetworksClassifier(SequenceClassifier):
    def __init__(self, name='Neural Networks', top_words=5000, max_review_length=500, embedding_vector_length=32,
                 memory_units=100, output_size=1, activation='sigmoid', loss_function='binary_crossentropy',
                 optimizer='adam', metrics=('accuracy',), epochs=3, batch_size=64):
        super(NeuralNetworksClassifier, self).__init__(name)
        self.max_review_length = max_review_length
        self.embedding_vector_length = embedding_vector_length
        self.top_words = top_words
        self.memory_units = memory_units
        self.output_size = output_size
        self.activation = activation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model_ = Sequential()
        self.model_.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.max_review_length))
        self.model_.add(LSTM(self.memory_units))
        self.model_.add(Dense(self.output_size, activation=self.activation))
        self.model_.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def get_transformer(self):
        return NeuralNetworksTransformer(self.max_review_length)


class NeuralNetworksTransformer(SequenceTransformer):
    def __init__(self, max_review_length=500):
        self.max_review_length = max_review_length

    def transform(self, X):
        # truncate and pad input sequences
        return sequence.pad_sequences(X, maxlen=self.max_review_length)

    def fit_transform(self, X):
        return self.transform(X)
