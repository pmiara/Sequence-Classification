# generic class must be extended
class SequenceClassifier:
    def __init__(self, name):
        self.name = name

    def fit(self):
        raise NotImplementedError

    def predict(self, predicting_set):
        raise NotImplementedError
