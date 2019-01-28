import pandas as pd
from prefixspan import PrefixSpan

from .sequence_classifier import SequenceClassifier


class PatternBasedClassifier(SequenceClassifier):
    def __init__(self, name='PatternBased', transformer=None, min_support=1, min_len=1, max_len=1000, k=10):
        super(PatternBasedClassifier, self).__init__(name, transformer)
        self.min_support = min_support
        self.min_len = min_len
        self.max_len = max_len
        self.k = k
        self.rules = None

    def _fit(self, X_train, y_train):
        df = pd.DataFrame({'X': X_train, 'y': y_train})
        df['y'] = y_train
        rules = df.groupby('y').apply(lambda rows: self._find_frequent_patterns(rows['X'].tolist()))
        rules['support_all'] = rules['pattern'].apply(lambda patt: self._count_support(df['X'], patt))
        rules['confidence'] = rules['support'] / rules['support_all']
        rules['length'] = rules['pattern'].apply(len)
        rules = rules.sort_values(['confidence', 'support', 'length'], ascending=False)
        self.rules = rules.reset_index(level=1, drop=True).reset_index()

    def _predict(self, X):
        result = []
        for x in X:
            rules_satisfying_condition = self.rules['pattern'].apply(lambda patt: self._contains_pattern(x, patt))
            try:
                first_rule = self.rules[rules_satisfying_condition].iloc[0]
                predicted_class = first_rule['y']
            except IndexError:
                predicted_class = self.rules['y'].iloc[0]
            result.append(predicted_class)
        return result

    def _find_frequent_patterns(self, X):
        ps = PrefixSpan(X)
        ps.min_len = self.min_len
        ps.max_len = self.max_len
        frequent_patterns = ps.topk(self.k,
            filter=lambda patt, matches: len(matches) > self.min_support)
        return pd.DataFrame(frequent_patterns, columns=['support', 'pattern'])

    @staticmethod
    def _contains_pattern(sequence, pattern):
        start = 0
        for symbol in pattern:
            if symbol in sequence[start:]:
                start += sequence[start:].index(symbol)
            else:
                return False
        return True

    @staticmethod
    def _count_support(X, pattern):
        return X.apply(lambda x: PatternBasedClassifier._contains_pattern(x, pattern)).sum()
