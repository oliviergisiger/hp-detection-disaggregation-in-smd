import numpy as np
import pandas as pd
import pickle

from domain import configs
from data_preprocessing.capabilities.data_augmentation import augment_seq
from data_preprocessing.capabilities.data_utils import remove_na
from data_preprocessing.capabilities.data_utils import get_auxilary_features, \
    get_temperature_features, get_temporal_features, get_load_features

np.random.seed(42)

DATA_FILENAME = configs.FILENAMES.DATA
IDS_FILENAME = configs.FILENAMES.IDS


class DisaggregationDataLoader:

    target = ['value_kwh']
    features = ['value_kwh_masked', 'temp', 'mean_temp', 'min_temp', 'min_of_day', 'dow']

    def __init__(self, sequence_length: int, augmentation_shift: int = None):
        self.sequence_length = sequence_length
        self.augmentation_shift = self._get_augmentation_shift(augmentation_shift)

    def create_train_test_data(self, path: str, id_col: str, write: bool = True, train_split=0.8):
        """
        interface to TSCDataLoader class. reads, transforms, augments and standardizes data.
        :param path: str. path to file
        :param id_col: str. column name that containes id
        :param write: bool. if train/test files should be written to pickle.
        :param train_split: float.
        :return: tuple with train, test data; x: 3D-tensor, y: 1D-array
        """

        train_ids, test_ids = self._get_ids(path, train_split=train_split)
        train_arrays, test_arrays = self._read_file(path, train_ids, test_ids)
        x_train, x_train_aux, y_train = self._prepare_data(train_arrays, augment_shift=self.augmentation_shift)
        x_test, x_test_aux, y_test = self._prepare_data(test_arrays, augment_shift=int(self.sequence_length/2))
        if write:
            _path = '../data/model_input/ckw/disaggregation/'
            self._write_file((x_train, x_train_aux, y_train), _path, 'train')
            self._write_file((x_test, x_test_aux, y_test), _path, 'test')
        return x_train, x_test, y_train, y_test

    def _read_file(self, path: str, train_ids: list, test_ids: list):
        """
        reads file at path and filters for ids.
        :param path: str. path to (pickle) file
        :param train_ids: list with train ids
        :param test_ids: list with test ids
        :return: tuple of lists of arrays, ready for self._prepare_data
        """

        cols = self.features + self.target
        df = pd.read_pickle(path + DATA_FILENAME)
        df_train = df[df.index.isin(train_ids)]
        df_test = df[df.index.isin(test_ids)]
        train = df_train.groupby(df_train.index)[cols].apply(np.array).values
        test = df_test.groupby(df_test.index)[cols].apply(np.array).values
        return train, test

    def _prepare_data(self, arrays, augment_shift: int = None):
        """
        splits input into equal length sequences. uses window slicing to augment data
        :param arrays: list of np.ndarrays
        :return: tuple X, y
        """
        xy = []
        aux = []
        for arr in arrays:
            features, target = arr[:, :-1], arr[:, -1]

            load = features[:, 0]
            temperature = features[:, 1]
            load_features = get_load_features(load)
            weather_features = get_temperature_features(temperature)
            temporal_features = get_temporal_features(features[:, -2])

            arr = np.column_stack([load, load_features, temperature, weather_features, temporal_features,
                                   target])

            arr = augment_seq(arr, self.sequence_length, self._get_augmentation_shift(augment_shift))
            arr = remove_na(arr)
            if arr.shape[0] == 0:
                continue
            _aux = get_auxilary_features(arr)
            xy.append(arr)
            aux.append(_aux)
        xy = np.concatenate(xy, axis=0)
        aux = np.concatenate(aux, axis=0)
        y = xy[:, :, -1]
        X = xy[:, :, :-1]
        return X, aux, y

    def _prepare_inference_data(self, path: str, train_split: float = 0.8):
        cols = self.features + self.target
        _, test_ids = self._get_ids(path, train_split)
        df = pd.read_pickle(path + DATA_FILENAME)
        df = df[df.index.isin(test_ids)]
        values = df.groupby(df.index)[cols].apply(np.array).values
        idx = df.groupby(df.index)[cols].apply(np.array).index

        x = []
        aux = []
        y = []
        for arr in values:
            features, target = arr[:, :-1], arr[:, -1]

            load = features[:, 0]
            temperature = features[:, 1]
            load_features = get_load_features(load)
            weather_features = get_temperature_features(temperature)
            temporal_features = get_temporal_features(features[:, -2])

            _xy = np.column_stack([load, load_features, temperature, weather_features, temporal_features,
                                   target])
            _aux = np.array([np.min(load), np.max(load), np.mean(load), np.var(load)])

            x.append(_xy[:, :-1])
            aux.append(_aux)
            y.append(_xy[:, -1])
        data = (idx, x, aux, y)
        with open(path + 'inference_data.p', 'wb') as f:
            pickle.dump(data, f)

    def _write_file(self, data: tuple, path: str, split: str):
        fname = f'{path}/ts_{self.sequence_length}_{self.augmentation_shift}_{split}'
        with open(f'{fname}.p', 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def _get_ids(path: str, train_split: float = 0.8) -> tuple:
        """
        split ids into train and test, keeping ratio
        :return: tuple train test ids
        """
        ids = pd.read_pickle(path + IDS_FILENAME)
        train_ids = ids.sample(frac=train_split, random_state=1)
        test_ids = ids[~ids.isin(train_ids)]
        print(train_ids.shape, test_ids.shape)
        return list(train_ids), list(test_ids)

    def _get_augmentation_shift(self, augmentation_shift):
        if augmentation_shift:
            return augmentation_shift
        return self.sequence_length


def validation_split(x, y, val_split=0.2):
    """
    splits x, y into separates arrays that can be used as train and validation split.
    """
    assert x.shape[0] == y.shape[0], 'features and labels must have the same length'
    int_val_split = int(val_split * x.shape[0])
    idx = np.random.choice(x.shape[0], int_val_split, replace=False)
    x_train, y_train = np.delete(x, idx, axis=0), np.delete(y, idx)
    x_val, y_val = x[idx], y[idx]
    return x_train, x_val, y_train, y_val


if __name__ == '__main__':  # ckw arr shape : (3, 1344, 4)
    dataloader = DisaggregationDataLoader(sequence_length=672, augmentation_shift=672)
    path = '../data/clean/ckw/disaggregation/'

    x_train, x_test, y_train, y_test = dataloader.create_train_test_data(path=path, id_col='id', train_split=0.47)
    print(x_train.shape, y_train.shape, y_train.sum())


