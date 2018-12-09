import editdistance
from keras.preprocessing import sequence
from sklearn.neighbors import KNeighborsClassifier

from .sequence_classifier import SequenceClassifier


class KNNClassifier(SequenceClassifier):
    def __init__(self, name='KNN', metric='editdistance', max_sequence_len=500, n_neighbors=3):
        super(KNNClassifier, self).__init__(name)
        if metric == 'editdistance':
            metric = editdistance.eval
        elif metric == 'longest_common_subsequence':
            metric = KNNClassifier.longest_common_subsequence_metric
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

    @staticmethod
    def longest_common_subsequence_metric(s1, s2):
        '''
        Source: https://rosettacode.org/wiki/Longest_common_subsequence#Python
        '''
        lengths = [[0 for j in range(len(s2)+1)] for i in range(len(s1)+1)]
        for i, x in enumerate(s1):
            for j, y in enumerate(s2):
                if x == y:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
        result = 0
        x, y = len(s1), len(s2)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert s1[x-1] == s2[y-1]
                result += 1
                x -= 1
                y -= 1
        return result
