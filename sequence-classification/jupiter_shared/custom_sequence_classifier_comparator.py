from sequence_transformer import SequenceTransformer
from sequence_classifier import SequenceClassifier
from sequence_classifier_comparator import SequenceClassifierComparator


# example
class CustomTransformer(SequenceTransformer):
    def __init__(self):
        SequenceTransformer.__init__(self)

    def transform(self, data):
        return data


class CustomClassifier(SequenceClassifier):
    def __init__(self, name):
        SequenceClassifier.__init__(self, name)

    def fit(self, fitting_set):
        pass

    def predict(self, predicting_set):
        result = []
        for p in predicting_set:
            if p == 1:
                result.append(11)
            else:
                result.append(p)
        return result


class CustomClassifier2(SequenceClassifier):
    def __init__(self, name):
        SequenceClassifier.__init__(self, name)

    def fit(self, fitting_set):
        pass

    def predict(self, predicting_set):
        result = []
        for p in predicting_set:
            result.append(p % 3)
        return result


class CustomComparator(SequenceClassifierComparator):
    def __init__(self):
        SequenceClassifierComparator.__init__(self)

    def score_classifiers(self):
        example = [11, 2, 3, 4, 5]
        for name, prediction in self.predictions:
            score = 0
            for i in range(len(prediction)):
                if prediction[i] == example[i]:
                    score += 1
            self.scores.append((name, score / len(example)))
        return self.scores


# custom_transformer = CustomTransformer()
# custom_classifier = CustomClassifier("custom1")
# custom_classifier2 = CustomClassifier2("custom2")
# custom_comparator = CustomComparator()
# custom_comparator.add_classifier(custom_classifier, custom_transformer)
# custom_comparator.add_classifier(custom_classifier2)
# custom_comparator.fit_classifiers([1, 2, 3, 4, 5])
# custom_comparator.predict_classifiers([1, 2, 3, 4, 5])
# custom_comparator.score_classifiers()
