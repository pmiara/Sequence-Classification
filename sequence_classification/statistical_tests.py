import Orange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, f_oneway, ttest_ind, shapiro, levene, wilcoxon, rankdata
from scikit_posthocs import posthoc_nemenyi_friedman


class StatisticalTests:

    def __init__(self, alpha, measurements, dataset_names, classifier_names, rounded=3):
        self.alpha = alpha
        self.measurements = measurements
        self.dataset_names = dataset_names
        self.classifier_names = classifier_names
        self.rounded = rounded

    def compare_on_datasets_separately(self):
        for dataset, dataset_name in zip(self.measurements, self.dataset_names):
            print('Tests for {} dataset'.format(dataset_name))
            self.compare_on_single_dataset(dataset)
            print('\n')

    def compare_on_single_dataset(self, dataset_measurements):
        if self.check_anova_assumptions(dataset_measurements):
            anova = self.calc_anova_p_value(dataset_measurements)
            if anova < self.alpha:
                print('Result of ANOVA test is negative: {} < {}'.format(anova, self.alpha))
                print('Classifiers on this dataset don\'t have the same expected value.\n')

                print('Results of t-Student test')
                t_student_results = self.calc_t_student(dataset_measurements)
                df = pd.DataFrame(t_student_results, columns=self.classifier_names, index=self.classifier_names)
                print(df)
            else:
                print('Result of ANOVA test is positive: {} >= {}'.format(anova, self.alpha))
                print('Classifiers on this dataset have the same expected value.')
        else:
            print('The assumptions of the ANOVA test are not met')
            # Kruskal-Wallis H-test could be run here (scipy.stats.kruskal)

    def check_anova_assumptions(self, dataset_measurements):
        if len(dataset_measurements[0]) == 1:
            return False
        if shapiro(dataset_measurements)[1] < self.alpha:
            return False
        if levene(*dataset_measurements)[1] < self.alpha:
            return False
        return True

    def compare_on_all_datasets(self):
        print('Tests for all datasets\n')
        if len(self.classifier_names) == 2:
            self.compare_with_wilcoxon()
        else:
            self.compare_with_friedman()

    def compare_with_wilcoxon(self):
        wilcoxon_result = self.calc_wilcoxon_p_value()
        if wilcoxon_result < self.alpha:
            print('Result of Wilcoxon test is negative: {} < {}'.format(wilcoxon_result, self.alpha))
            print('Classifiers don\'t come from the same distribution')
        else:
            print('Result of Wilcoxon test is positive: {} >= {}'.format(wilcoxon_result, self.alpha))
            print('Classifiers come from the same distribution')

    def compare_with_friedman(self):
        friedman = self.calc_friedman_p_value()
        if friedman < self.alpha:
            print('Result of Friedman test is negative: {} < {}'.format(friedman, self.alpha))
            print('Classifiers are not the same\n')

            print('Results of Nemenyi test')
            print('Classifiers with their average ranks')
            avg_ranks = np.mean([rankdata(-a) for a in self.calc_averages()], axis=0)
            for classifier_name, avg_rank in zip(self.classifier_names, avg_ranks):
                print('- {}: {}'.format(classifier_name, round(avg_rank, self.rounded)))

            cd = self.calc_critical_distance(avg_ranks)
            print('Critical distance: {}'.format(cd))
            Orange.evaluation.graph_ranks(avg_ranks, self.classifier_names, cd=cd, width=6, textspace=1.5)
            plt.show()
        else:
            print('Result of Friedman test is positive: {} >= {}'.format(friedman, self.alpha))
            print('Classifiers are the same')


    def calc_anova_p_value(self, dataset_measurements):
        anova = f_oneway(*dataset_measurements)[1]
        return round(anova, self.rounded)

    def calc_t_student(self, dataset_measurements):
        t_student = []
        for classifier1 in dataset_measurements:
            classifier_result = []
            for classifier2 in dataset_measurements:
                classifier_result.append(ttest_ind(classifier1, classifier2)[1])
            t_student.append(classifier_result)
        return np.round(t_student, self.rounded)

    def calc_wilcoxon_p_value(self):
        averages = self.calc_averages()
        first, second = zip(*averages)
        return round(wilcoxon(first, second)[1], self.rounded)

    def calc_friedman_p_value(self):
        averages = self.calc_averages()
        friedman = friedmanchisquare(*averages)[1]
        return round(friedman, self.rounded)

    def calc_nemenyi(self):
        averages = self.calc_averages()
        nemenyi = posthoc_nemenyi_friedman(averages)
        return np.round(nemenyi, self.rounded)

    def calc_averages(self):
        return [np.average(dataset, axis=1) for dataset in self.measurements]

    def calc_critical_distance(self, avg_ranks):
        cd = Orange.evaluation.compute_CD(avg_ranks, len(self.dataset_names))
        return round(cd, self.rounded)
