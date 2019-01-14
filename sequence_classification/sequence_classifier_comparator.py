from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from .results_presenter import ResultsPresenter
from .datasets.utils import Dataset


class SequenceClassifierComparator:
    def __init__(self, writer, reader, classifiers_with_params=None, cv=3):
        if classifiers_with_params is None:
            classifiers_with_params = []
        self.classifiers_with_params = classifiers_with_params
        self.writer = writer
        self.reader = reader
        self.datasets = []
        self.cv = cv

    def add_classifier(self, classifier, params=None):
        if params is None:
            params = {}
        self.classifiers_with_params.append((classifier, params))

    def add_dataset(self, loader, name=None):
        if name:
            dataset = loader.load_data(name)
        else:
            dataset = loader.load_data()
        self.datasets.append(dataset)

    def add_custom_dataset(self, X, y, name):
        self.datasets.append(Dataset(X, y, name))

    def fit_predict_all(self, split_params=None, rounds=3):
        if split_params is None:
            split_params = {}
        for dataset in self.datasets:
            for i in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, **split_params)
                for classifier, params in self.classifiers_with_params:
                    print('{}, round {}, with {}-fold cross validation'.format(classifier.name, i + 1, self.cv))
                    results = self.fit_predict(X_train, y_train, X_test, y_test, classifier, params)
                    self.writer.write_results(dataset.name, classifier.name, *results)

    def fit_predict(self, X_train, y_train, X_test, y_test, classifier, params):
        grid = GridSearchCV(classifier, params, cv=self.cv, scoring='accuracy')
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        classifier.set_params(**best_params)
        classifier.fit(X_train, y_train)

        y_pred_train = classifier.predict(X_train)
        conf_matrix_train = confusion_matrix(y_train, y_pred_train)
        y_pred_test = classifier.predict(X_test)
        conf_matrix_test = confusion_matrix(y_test, y_pred_test)
        return best_params, conf_matrix_train, conf_matrix_test

    def get_presenter(self):
        dataset_names = [d.name for d in self.datasets]
        classifier_names = [c[0].name for c in self.classifiers_with_params]
        results = self.reader.read_results(dataset_names, classifier_names)
        return ResultsPresenter(results)
