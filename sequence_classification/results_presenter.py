import itertools
import numpy as np
import matplotlib.pyplot as plt


FONT_SIZE = 'x-large'

class ResultsPresenter:
    """
    Results contain confusion matrices and best parameters for each classifier for each round of training.
    There are confusion matrices for both train set and test set.
    """
    def __init__(self, results):
        self.results = results

    def show_all(self):
        self.show_confusion_matrices()
        self.show_box_plots()

    def show_confusion_matrices(self):
        for dataset in self.results:
            print(dataset)
            for classifier_name, values in self.results[dataset].items():
                conf_mat_train = np.mean([v['conf_matrix_train'] for v in values], axis=0).astype('int')
                conf_mat_test = np.mean([v['conf_matrix_test'] for v in values], axis=0).astype('int')
                no_of_classes = len(conf_mat_train)
                plt.figure(figsize=(15, 15))
                plt.subplot(1, 2, 1)
                self.plot_confusion_matrix(conf_mat_train, classes=[str(i) for i in range(no_of_classes)],
                                           title='Train confusion matrix for {}'.format(classifier_name))
                plt.subplot(1, 2, 2)
                self.plot_confusion_matrix(conf_mat_test, classes=[str(i) for i in range(no_of_classes)],
                                           title='Test confusion matrix for {}'.format(classifier_name))
                plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues, font_size=FONT_SIZE):
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
            color = 'white' if cm[i, j] > thresh else 'black'
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                     fontsize=font_size, color=color)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def show_box_plots(self):
        accuracies = {'train': [], 'test': []}
        names = []
        for dataset in self.results:
            print(dataset)
            for classifier_name, values in self.results[dataset].items():
                accuracies['train'].append([self.calc_accuracy_from_cm(v['conf_matrix_train']) for v in values])
                accuracies['test'].append([self.calc_accuracy_from_cm(v['conf_matrix_test']) for v in values])
                names.append(classifier_name)

            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            plt.title('Accuracy on train set', fontsize=FONT_SIZE)
            plt.boxplot(accuracies['train'], labels=names)
            plt.subplot(1, 2, 2)
            plt.title('Accuracy on test set', fontsize=FONT_SIZE)
            plt.boxplot(accuracies['test'], labels=names)
            plt.show()

    @staticmethod
    def calc_accuracy_from_cm(cm):
        return cm.diagonal().sum() / cm.sum()
