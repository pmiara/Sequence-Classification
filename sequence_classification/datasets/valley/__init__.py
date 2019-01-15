import re

from ..utils import DatasetLoader, Dataset, get_X_y


class ValleyDataset(DatasetLoader):

    @staticmethod
    def load_data(name='Valley'):
        valley_data = []
        with open(ValleyDataset.get_dataset_file(['valley', 'valley_data.txt']), 'r') as f:
            file_content = f.readlines()
            for line in file_content:
                values = line.split('\t')
                valley_data.append((str(values[0]), int(re.sub('\D', '', values[1]))))
        X, y = get_X_y(valley_data)

        return Dataset(X, y, name)
