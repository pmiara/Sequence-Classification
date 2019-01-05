from sequence_classification.results_presenter import ResultsPresenter
import pytest
import numpy as np
from scipy.stats import friedmanchisquare

@pytest.fixture
def prepare_data():
    data = {'small_imdb':
             {'SVM': [{'conf_matrix_train': np.array([[370,   0],
                       [  0, 380]]), 'conf_matrix_test': np.array([[100,  36],
                       [ 22,  92]]), 'params': {}},
               {'conf_matrix_train': np.array([[382,   0],
                       [  0, 368]]), 'conf_matrix_test': np.array([[ 98,  26],
                       [ 17, 109]]), 'params': {}}],
              'kk': [{'conf_matrix_train': np.array([[370,   0],
                       [  0, 380]]), 'conf_matrix_test': np.array([[100,  36],
                       [ 22,  92]]), 'params': {}},
               {'conf_matrix_train': np.array([[382,   0],
                       [  0, 368]]), 'conf_matrix_test': np.array([[ 98,  26],
                       [ 17, 109]]), 'params': {}}],
              'HMM': [{'conf_matrix_train': np.array([[348,  22],
                       [  4, 376]]), 'conf_matrix_test': np.array([[ 67,  69],
                       [  8, 106]]), 'params': {}},
               {'conf_matrix_train': np.array([[332,  50],
                       [  4, 364]]), 'conf_matrix_test': np.array([[ 46,  78],
                       [ 13, 113]]), 'params': {}}]},
 'small_imdb1':
             {'SVM': [{'conf_matrix_train': np.array([[372,   0],
                       [  0, 378]]), 'conf_matrix_test': np.array([[107,  27],
                       [ 25,  91]]), 'params': {}},
               {'conf_matrix_train': np.array([[386,   0],
                       [  0, 364]]), 'conf_matrix_test': np.array([[101,  19],
                       [ 30, 100]]), 'params': {}}],
              'kk': [{'conf_matrix_train': np.array([[372,   0],
                       [  0, 378]]), 'conf_matrix_test': np.array([[107,  27],
                       [ 25,  91]]), 'params': {}},
               {'conf_matrix_train': np.array([[386,   0],
                       [  0, 364]]), 'conf_matrix_test': np.array([[101,  19],
                       [ 30, 100]]), 'params': {}}],
              'HMM': [{'conf_matrix_train': np.array([[364,   8],
                       [  9, 369]]), 'conf_matrix_test': np.array([[ 88,  46],
                       [ 16, 100]]), 'params': {}},
               {'conf_matrix_train': np.array([[385,   1],
                       [ 55, 309]]), 'conf_matrix_test': np.array([[113,   7],
                       [ 69,  61]]), 'params': {}}]},
 'small_imdb2':
             {'SVM': [{'conf_matrix_train': np.array([[388,   0],
                       [  0, 362]]), 'conf_matrix_test': np.array([[ 94,  24],
                       [ 30, 102]]), 'params': {}},
               {'conf_matrix_train': np.array([[377,   0],
                       [  0, 373]]), 'conf_matrix_test': np.array([[101,  28],
                       [ 21, 100]]), 'params': {}}],
              'kk': [{'conf_matrix_train': np.array([[388,   0],
                       [  0, 362]]), 'conf_matrix_test': np.array([[ 94,  24],
                       [ 30, 102]]), 'params': {}},
               {'conf_matrix_train': np.array([[377,   0],
                       [  0, 373]]), 'conf_matrix_test': np.array([[101,  28],
                       [ 21, 100]]), 'params': {}}],
              'HMM': [{'conf_matrix_train': np.array([[384,   4],
                       [ 27, 335]]), 'conf_matrix_test': np.array([[92, 26],
                       [42, 90]]), 'params': {}},
               {'conf_matrix_train': np.array([[367,  10],
                       [  4, 369]]), 'conf_matrix_test': np.array([[ 86,  43],
                       [ 20, 101]]), 'params': {}}]}}
    return ResultsPresenter(data)



def test_should_correct_friedman(prepare_data):
    result = prepare_data.calc_friedman_value()
    assert round(result[0], 3) == 0.8
    assert round(result[1], 3) == 0.67


def test_should_correct_anova(prepare_data):
    result = prepare_data.calc_anova_value()
    assert round(result[0][0], 3) == 6.949
    assert round(result[0][1], 3) == 0.075
    assert len(result) == 3