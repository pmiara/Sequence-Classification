from ..utils import DatasetLoader, Dataset, get_X_y


class BioDataset(DatasetLoader):

    @staticmethod
    def load_data(name='BioData'):
        bio_data = []
        with open(BioDataset.get_dataset_file(['bio_data', 'bioData.data']), 'r') as f:
            lines = f.readlines()
            for line in lines:
                bio_class, _, sequence = line.replace(' ', '').replace('\n', '').split(',')
                bio_data.append((' '.join(list(sequence)), bio_class))
        X, y = get_X_y(bio_data)
        return Dataset(X, y, name)
