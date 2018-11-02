# generic class must be extended
class GenericSequenceClassifier:
    def __init__(self, name):
        self.name = name

    def fit(self):
        pass

    def predict(self, predicting_set):
        pass
