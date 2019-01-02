from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

from .sequence_classifier import SequenceClassifier
from sequence_transformer import SequenceTransformer


class SVMClassifier(SequenceClassifier):
    def __init__(self, name='SVM'):
        super(SVMClassifier, self).__init__(name)
        self.model = LinearSVC(random_state=0, tol=1e-5)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_transformer(self):
        return SVMTransformer()


class SVMTransformer(SequenceTransformer):
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 token_pattern='\\b\\w+\\b')

    def fit_transform(self, raw_data):
        strings_from_list = [' '.join([str(x) for x in data]) for data in raw_data]
        self.vectorizer.fit_transform(strings_from_list)
        return self.vectorizer.fit_transform(strings_from_list).toarray()

    def transform(self, raw_data):
        strings_from_list = [' '.join([str(x) for x in data]) for data in raw_data]
        return self.vectorizer.transform(strings_from_list).toarray()



