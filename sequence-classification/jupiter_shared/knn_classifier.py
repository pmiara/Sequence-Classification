from sequence_classifier import SequenceClassifier
from keras.preprocessing import sequence
from sklearn.neighbors import KNeighborsClassifier
import editdistance

from sequence_transformer import SequenceTransformer


class KNNClassifier(SequenceClassifier):
    def __init__(self, name='KNN', metric='editdistance', max_sequence_len=500, n_neighbors=3):
        super(KNNClassifier, self).__init__(name)
        if metric == 'editdistance':
            metric = editdistance.eval
        self.metric = metric
        self.max_sequence_len = max_sequence_len
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.model_ = KNeighborsClassifier(
            metric=self.metric, n_neighbors=self.n_neighbors)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def get_transformer(self):
        return KNNTransformer(self.max_sequence_len)


class KNNTransformer(SequenceTransformer):

    def __init__(self, max_sequence_len):
        self.max_sequence_len = max_sequence_len

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return sequence.pad_sequences(X, maxlen=self.max_sequence_len)
