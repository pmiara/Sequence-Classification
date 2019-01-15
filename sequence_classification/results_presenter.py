import itertools
import numpy as np
import matplotlib.pyplot as plt

from .statistical_tests import StatisticalTests


FONT_SIZE = 'x-large'
DEFAULT_RESULTS_TYPE = 'test'

class ResultsPresenter:
    """
    Results contain confusion matrices and best parameters for each classifier for each round of training.
    There are confusion matrices for both train set and test set.
    """

    def __init__(self, results):
        self.results = results

    def show_all(self, results_type=DEFAULT_RESULTS_TYPE):
        self.show_all_confusion_matrices(results_type)
        self.show_box_plots(results_type)
        statistical_tests = self.get_statistical_tests(alpha=0.05, results_type=results_type)
        statistical_tests.compare_on_datasets_separately()
        statistical_tests.compare_on_all_datasets()

    def show_all_confusion_matrices(self, results_type=DEFAULT_RESULTS_TYPE):
        for dataset in self.results:
            self.show_confusion_matrices_for_dataset(dataset, results_type)

    def show_confusion_matrices_for_dataset(self, dataset, results_type=DEFAULT_RESULTS_TYPE):
        for classifier_name in self.results[dataset]:
            self.show_confusion_matrix(dataset, classifier_name, results_type)

    def show_confusion_matrix(self, dataset, classifier_name, results_type=DEFAULT_RESULTS_TYPE):
        results_type = self.get_results_type(results_type)
        values = self.results[dataset][classifier_name]
        confusion_matrix = np.mean([v[results_type] for v in values], axis=0).astype('int')
        self.plot_confusion_matrix(confusion_matrix,
                                   title='Confusion matrix for {} on {}'.format(classifier_name, dataset))

    @staticmethod
    def plot_confusion_matrix(cm, title, normalize=False, cmap=plt.cm.Blues, font_size=FONT_SIZE):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            sums_in_rows = cm.sum(axis=1)[:, np.newaxis]
            cm = cm / sums_in_rows

        no_of_classes = len(cm)
        classes = [str(i) for i in range(no_of_classes)]

        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=font_size)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > thresh else 'black'
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                     fontsize=font_size, color=color)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def show_box_plots(self, results_type=DEFAULT_RESULTS_TYPE):
        for dataset in self.results:
            self.show_box_plot(dataset, results_type)

    def show_box_plot(self, dataset, results_type=DEFAULT_RESULTS_TYPE):
        results_type = self.get_results_type(results_type)
        accuracies, names = [], []

        for classifier_name, values in self.results[dataset].items():
            accuracies.append([self.calc_accuracy_from_cm(v[results_type]) for v in values])
            names.append(classifier_name)

        plt.figure(figsize=(7, 7))
        plt.title('Accuracy on {}'.format(dataset), fontsize=FONT_SIZE)
        plt.boxplot(accuracies, labels=names)
        plt.show()

    def get_statistical_tests(self, alpha, round_precision=3, results_type=DEFAULT_RESULTS_TYPE):
        results_type = self.get_results_type(results_type)
        dataset_names = [dataset for dataset in self.results]
        classifier_names = [classifier for classifier in self.results[dataset_names[0]]]
        return StatisticalTests(alpha, self._prepare_measurements(results_type), dataset_names, classifier_names, round_precision)

    def _prepare_measurements(self, results_type='conf_matrix_test'):
        if not self.results:
            raise AttributeError('self.results is not defined')
        measurements = []
        for dataset_result in self.results.values():
            dataset_measurements = []
            for classifier_name, rounds in dataset_result.items():
                classifier_measurements = []
                for r in rounds:
                    acc = self.calc_accuracy_from_cm(r[results_type])
                    classifier_measurements.append(acc)
                dataset_measurements.append(classifier_measurements)
            measurements.append(dataset_measurements)
        return measurements

    @staticmethod
    def calc_accuracy_from_cm(cm):
        return cm.diagonal().sum() / cm.sum()

    @staticmethod
    def get_results_type(results_type):
        if results_type not in ['train', 'test']:
            raise ValueError('results_type should be equal to "train" or "test"')
        results_type = 'conf_matrix_' + results_type
        return results_type
