from sequence_classifier import SequenceClassifier
from keras.preprocessing import sequence
from sklearn.neighbors import KNeighborsClassifier
import editdistance



class KNNClassifier(SequenceClassifier):
    def __init__(self, name='KNN', metric='editdistance', max_sequence_len=500, n_neighbors=3):
        super(KNNClassifier, self).__init__(name)
        if metric == 'editdistance':
            metric = editdistance.eval
        self.metric = metric
        self.max_sequence_len = max_sequence_len
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X = sequence.pad_sequences(X, maxlen=self.max_sequence_len)
        self.model_ = KNeighborsClassifier(
            metric=self.metric, n_neighbors=self.n_neighbors)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = sequence.pad_sequences(X, maxlen=self.max_sequence_len)
        return self.model_.predict(X)
