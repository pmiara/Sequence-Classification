from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from .results_presenter import ResultsPresenter
from .datasets.utils import Dataset


class SequenceClassifierComparator:
    def __init__(self, writer, reader, classifier_triplets=None, cv=3):
        if classifier_triplets is None:
            classifier_triplets = []
        self.classifier_triplets = classifier_triplets
        self.writer = writer
        self.reader = reader
        self.datasets = []
        self.cv = cv

    def add_classifier(self, classifier, params=None, sequence_transformer=None):
        if params is None:
            params = {}
        self.classifier_triplets.append((classifier, params, sequence_transformer))

    def add_dataset(self, loader, name=None):
        if name:
            dataset = loader.load_data(name)
        else:
            dataset = loader.load_data()
        self.datasets.append(dataset)

    def add_other_dataset(self, X, y, name):
        self.datasets.append(Dataset(X, y, name))

    def fit_predict_all(self, split_params=None, rounds=3):
        if split_params is None:
            split_params = {}
        for dataset in self.datasets:
            for i in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, **split_params)
                for classifier, params, transformer in self.classifier_triplets:
                    print('{}, round {}, with {}-fold cross validation'.format(classifier.name, i + 1, self.cv))
                    X_train_transform, X_test_transform = self.apply_transformer(X_train, X_test, transformer)
                    results = self.fit_predict(X_train_transform, y_train, X_test_transform, y_test, classifier, params)
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

    @staticmethod
    def apply_transformer(X_train, X_test, transformer):
        if transformer is not None:
            X_train_transform = transformer.fit_transform(X_train)
            X_test_transform = transformer.transform(X_test)
        else:
            X_train_transform = X_train
            X_test_transform = X_test
        return X_train_transform, X_test_transform

    def get_presenter(self):
        dataset_names = [d.name for d in self.datasets]
        classifier_names = [c[0].name for c in self.classifier_triplets]
        results = self.reader.read_results(dataset_names, classifier_names)
        return ResultsPresenter(results)
