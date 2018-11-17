# generic class must be extended
class SequenceTransformer:
    def transform(self, data):
        raise NotImplementedError

    def fit_transform(self, data):
        raise NotImplementedError
