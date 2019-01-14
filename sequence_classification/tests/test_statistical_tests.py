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
    return StatisticalTests(measurements)


def test_correct_anova(statistical_tests):
    result = statistical_tests.calc_anova_value()
    assert round(result[0][0], 3) == 6.949
    assert round(result[0][1], 3) == 0.075
    assert len(result) == 3

def test_correct_t_student(statistical_tests):
    result = statistical_tests.calc_t_student()
    assert len(result) == 3
    assert len(result[0]) == 3
    assert result[0][0][0][0] == 0
    assert result[0][0][0][1] == 1
    assert round(result[0][0][2][0], 3) == 3.265
    assert round(result[0][0][2][1], 3) == 0.082

def test_correct_friedman(statistical_tests):
    result = statistical_tests.calc_friedman_value()
    assert round(result[0], 3) == 0.8
    assert round(result[1], 3) == 0.67

def test_correct_nemenyi(statistical_tests):
    result = statistical_tests.calc_nemenyi()
    expected = [
        [-1, 0.9, 0.158],
        [0.9, -1, 0.158],
        [0.158, 0.158, -1]
    ]
    assert (np.round(result.values, 3) == expected).all()
