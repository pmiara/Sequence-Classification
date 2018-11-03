# generic class must be extended
class SequenceClassifierComparator:
    def __init__(self, classifiers=[]):
        self.classifiers = classifiers
        self.predictions = []
        self.scores = []

    def add_classifier(self, classifier, sequence_transformer=None):
        self.classifiers.append((classifier, sequence_transformer))

    def fit_classifiers(self, fitting_set):
        for classifier, transformer in self.classifiers:
            if transformer is not None:
                classifier.fit(transformer.transform(fitting_set))
            else:
                classifier.fit(fitting_set)

    def predict_classifiers(self, predicting_set):
        for classifier in self.classifiers:
            self.predictions.append((classifier[0].name, classifier[0].predict(predicting_set)))

    def score_classifiers(self):
        raise NotImplementedError

    def plot_comparison(self):
        raise NotImplementedError
