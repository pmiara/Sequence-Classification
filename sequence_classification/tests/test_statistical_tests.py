import pytest
import numpy as np

from sequence_classification.statistical_tests import StatisticalTests


@pytest.fixture
def statistical_tests():
    measurements = [
        [[0.768, 0.828], [0.768, 0.828], [0.692, 0.636]],
        [[0.792, 0.804], [0.792, 0.804], [0.752, 0.696]],
        [[0.784, 0.804], [0.784, 0.804], [0.728, 0.748]]
    ]
    return StatisticalTests(0.05, measurements, dataset_names=['a', 'b', 'c'], classifier_names=['d', 'e', 'f'])


def test_correct_anova(statistical_tests):
    dataset_measurements = statistical_tests.measurements[0]
    result = statistical_tests.calc_anova_p_value(dataset_measurements)
    assert result == 0.075

def test_correct_round(statistical_tests):
    statistical_tests.rounded = 2
    dataset_measurements = statistical_tests.measurements[0]
    result = statistical_tests.calc_anova_p_value(dataset_measurements)
    assert result == 0.07


def test_correct_t_student(statistical_tests):
    dataset_measurements = statistical_tests.measurements[0]
    result = statistical_tests.calc_t_student(dataset_measurements)
    assert len(result) == 3
    assert len(result[0]) == 3
    assert result[0][0] == 1
    assert result[0][2] == 0.082

def test_correct_friedman(statistical_tests):
    result = statistical_tests.calc_friedman_p_value()
    assert result == 0.67

def test_correct_nemenyi(statistical_tests):
    result = statistical_tests.calc_nemenyi()
    expected = [
        [-1, 0.9, 0.158],
        [0.9, -1, 0.158],
        [0.158, 0.158, -1]
    ]
    assert (result.values == expected).all()
