# generic class must be extended
class SequenceTransformer:
    def __init__(self):
        pass

    def transform(self, data):
        raise NotImplementedError
