import json

from ..utils import DatasetLoader, Dataset, get_X_y


class TrollDataset(DatasetLoader):

    @staticmethod
    def load_data(name='Troll'):
        troll_data = []
        with open(TrollDataset.get_dataset_file(['troll_detection', 'trollDetection.json']), 'r') as f:
            json_lines = f.readlines()
            for line in json_lines:
                json_line = json.loads(line)
                troll_data.append((json_line['content'], int(json_line['annotation']['label'][0])))

        X_troll, y_troll = get_X_y(troll_data)
        return Dataset(X_troll, y_troll, name)
