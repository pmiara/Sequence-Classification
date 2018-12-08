from sequence_transformer import SequenceTransformer
from sequence_classifier import SequenceClassifier
from sequence_classifier_comparator import SequenceClassifierComparator
from results_writer import ResultsWriter
from results_reader import ResultsReader


class CustomTransformer(SequenceTransformer):
    def __init__(self):
        SequenceTransformer.__init__(self)

    def transform(self, data):
        return data

    def fit_transform(self, data):
        return data


class CustomClassifier(SequenceClassifier):
    def __init__(self, name):
        SequenceClassifier.__init__(self, name)

    def fit(self, X, y):
        pass

    def predict(self, X):
        result = []
        for p in X:
            if p == 1:
                result.append(11)
            else:
                result.append(p)
        return result


class CustomClassifier2(SequenceClassifier):
    def __init__(self, name):
        SequenceClassifier.__init__(self, name)

    def fit(self, X, y):
        pass

    def predict(self, X):
        result = []
        for p in X:
            result.append(p % 3)
        return result


if __name__ == '__main__':
    custom_transformer = CustomTransformer()
    custom_classifier = CustomClassifier('custom1')
    custom_classifier2 = CustomClassifier2('custom2')
    comparator = SequenceClassifierComparator(ResultsWriter(), ResultsReader())
    comparator.add_classifier(custom_classifier, sequence_transformer=custom_transformer)
    comparator.add_classifier(custom_classifier2)
    comparator.fit_predict([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    presenter = comparator.get_presenter()
    presenter.show_all()
