import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


class SequenceClassifierComparator:
    def __init__(self, writer, reader, classifier_triplets=None):
        if classifier_triplets is None:
            classifier_triplets = []
        self.classifier_triplets = classifier_triplets
        self.writer = writer
        self.reader = reader

    def add_classifier(self, classifier, params=None, sequence_transformer=None):
        if params is None:
            params = {}
        self.classifier_triplets.append((classifier, params, sequence_transformer))

    def fit_predict(self, X, y, split_params=None, rounds=3, cv=3):
        if split_params is None:
            split_params = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        for classifier, params, transformer in self.classifier_triplets:
            X_train_transform, X_pred_transform = self.transform_data(X_train, X_test, transformer)
            for _ in range(rounds):
                grid = GridSearchCV(classifier, params, cv=cv)
                grid.fit(X_train_transform, y_train)

                best_params = grid.best_params_
                for param in best_params:
                    setattr(classifier, param, best_params[param])
                classifier.fit(X_train_transform, y_train)

                y_pred_train = classifier.predict(X_train)
                conf_matrix_train = confusion_matrix(y_train, y_pred_train)
                y_pred_test = classifier.predict(X_test)
                conf_matrix_test = confusion_matrix(y_test, y_pred_test)

                self.writer.write_results(classifier.name, best_params, conf_matrix_train, conf_matrix_test)

    @staticmethod
    def transform_data(X_train, X_test, transformer):
        if transformer is not None:
            X_train_transform = transformer.fit_transform(X_train)
            X_pred_transform = transformer.transform(X_test)
        else:
            X_train_transform = X_train
            X_pred_transform = X_test
        return X_train_transform, X_pred_transform

    def plot_comparison(self):
        classifier_names = [c[0].name for c in self.classifier_triplets]
        results = self.reader.read_results(classifier_names)
        for name, values in results:
            print("--------------")
            print(name)
            print(values["params"])
            fig, ax = plot_confusion_matrix(conf_mat=values["conf_matrix_test"])
            plt.show()
