# generic class must be extended
class SequenceClassifier:
    def __init__(self, name):
        self.name = name

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
