from sequence_classifier import SequenceClassifier
from keras.preprocessing import sequence

from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(SequenceClassifier):
    def __init__(self, name='KNN', distance_metric='hamming', max_sequence_len=500, **params):
        super(KNNClassifier, self).__init__(name)
        self.distance_metric = distance_metric
        self.max_sequence_len = max_sequence_len
        self.params = params

    def fit(self, X, y):
        X = sequence.pad_sequences(X, maxlen=self.max_sequence_len)
        self.model_ = KNeighborsClassifier(
            metric=self.distance_metric, **self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = sequence.pad_sequences(X, maxlen=self.max_sequence_len)
        return self.model_.predict(X)
