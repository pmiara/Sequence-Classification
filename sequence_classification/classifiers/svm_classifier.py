from sklearn.svm import LinearSVC

from .sequence_classifier import SequenceClassifier


class SVMClassifier(SequenceClassifier):
    def __init__(self, name='SVM', transformer=None):
        super(SVMClassifier, self).__init__(name, transformer)
        self.model = LinearSVC(random_state=0, tol=1e-5)

    def _fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def _predict(self, X):
        return self.model.predict(X)
