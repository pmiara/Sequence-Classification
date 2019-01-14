import numpy as np
from scipy.stats import friedmanchisquare, f_oneway, ttest_ind
from scikit_posthocs import posthoc_nemenyi_friedman


class StatisticalTests:

    def __init__(self, measurements):
        self.measurements = measurements

    def calc_averages(self):
        return [np.average(dataset, axis=1) for dataset in self.measurements]

    def calc_anova_value(self):
        return [f_oneway(*dataset) for dataset in self.measurements]

    def calc_t_student(self):
        t_student = []
        for dataset in self.measurements:
            dataset_result = []
            for classifier1 in dataset:
                classifier_result = []
                for classifier2 in dataset:
                    classifier_result.append(ttest_ind(classifier1, classifier2))
                dataset_result.append(classifier_result)
            t_student.append(dataset_result)
        return t_student

    def calc_friedman_value(self):
        averages = self.calc_averages()
        return friedmanchisquare(*averages)

    def calc_nemenyi(self):
        averages = self.calc_averages()
        return posthoc_nemenyi_friedman(averages)
