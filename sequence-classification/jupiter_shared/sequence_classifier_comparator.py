import itertools
import numpy as np
import matplotlib.pyplot as plt
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
            for i in range(rounds):
                print('{}, round {}, with {}-fold cross validation'.format(classifier.name, i + 1, cv))
                grid = GridSearchCV(classifier, params, cv=cv, scoring='accuracy')
                grid.fit(X_train_transform, y_train)

                best_params = grid.best_params_
                classifier.set_params(**best_params)
                classifier.fit(X_train_transform, y_train)

                y_pred_train = classifier.predict(X_train_transform)
                conf_matrix_train = confusion_matrix(y_train, y_pred_train)
                y_pred_test = classifier.predict(X_pred_transform)
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

    def show_results(self):
        classifier_names = [c[0].name for c in self.classifier_triplets]
        results = self.reader.read_results(classifier_names)
        accuracies, names = [], []
        for name, values in results.items():
            accuracies.append([self.calc_accuracy_from_cm(v['conf_matrix_test']) for v in values])
            names.append(name)
            conf_mat_train = np.mean([v['conf_matrix_train'] for v in values], axis=0).astype('int')
            conf_mat_test = np.mean([v['conf_matrix_test'] for v in values], axis=0).astype('int')
            no_of_classes = len(conf_mat_train)
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            self.plot_confusion_matrix(conf_mat_train, classes=[str(i) for i in range(no_of_classes)],
                                       title='Train confusion matrix for {}'.format(name))
            plt.subplot(1, 2, 2)
            self.plot_confusion_matrix(conf_mat_test, classes=[str(i) for i in range(no_of_classes)],
                                       title='Test confusion matrix for {}'.format(name))
            plt.show()
        self.show_boxplots(accuracies, names)

    @staticmethod
    def calc_accuracy_from_cm(cm):
        return cm.diagonal().sum() / cm.sum()

    def show_boxplots(self, accuracies, names):
        _, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Accuracy on test set', fontsize='x-large')
        ax.boxplot(accuracies, labels=names)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',
                              cmap=plt.cm.Blues, font_size='x-large'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=font_size)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center', fontsize=font_size,
                     color='white' if cm[i, j] > thresh else 'black')

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
