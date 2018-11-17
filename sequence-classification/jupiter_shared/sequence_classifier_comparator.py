from sklearn.metrics import accuracy_score
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


class SequenceClassifierComparator:
    def __init__(self, classifiers=None):
        if classifiers is None:
            classifiers = []
        self.classifiers = classifiers
        self.predictions = []
        self.scores = []

    def add_classifier(self, classifier, sequence_transformer=None):
        self.classifiers.append((classifier, sequence_transformer))

    def fit_classifiers(self, X_train, y_train):
        for classifier, transformer in self.classifiers:
            if transformer is not None:
                classifier.fit(transformer.transform(X_train), y_train)
            else:
                classifier.fit(X_train, y_train)

    def predict_classifiers(self, X):
        self.predictions = []
        for classifier, transformer in self.classifiers:
            if transformer is not None:
                y_pred = classifier.predict(transformer.transform(X))
            else:
                y_pred = classifier.predict(X)
            self.predictions.append((classifier.name, y_pred))
        return self.predictions

    def score_classifiers(self, X_test, y_test, results_writer):
        for name, y_pred in self.predict_classifiers(X_test):
            y_pred = [0 if i < 0.5 else 1 for i in y_pred]
            accuracy = accuracy_score(y_test, y_pred)
            matrix = confusion_matrix(y_test, y_pred)
            params = {"warstwy": 2, "koza": 3, "cztery": 4}
            results_writer.write_results(name,params,matrix, matrix)
            self.scores.append((name, accuracy))
        return self.scores

    def plot_comparison(self, results_reader):
        classifier_names = [c[0].name for c in self.classifiers]
        results = results_reader.read_results(classifier_names)
        for name, values in results:
            print("--------------")
            print(name)
            print(values["params"])
            fig, ax = plot_confusion_matrix(conf_mat=values["conf_matrix_test"])
            plt.show()
