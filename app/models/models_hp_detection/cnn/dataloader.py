import numpy as np
import pandas as pd
import pickle

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from domain import configs
from data_preprocessing.capabilities.data_augmentation import augment_seq
from data_preprocessing.capabilities.data_utils import remove_na, unison_shuffled_copies
from data_preprocessing.capabilities.data_utils import standardize, differentiate

np.random.seed(42)

DATA_FILENAME = configs.FILENAMES.DATA
IDS_FILENAME = configs.FILENAMES.IDS


class TSCDataLoader:
    _sampling_methods = ('over-sampling', 'under-sampling', 'unbalanced')

    target = ['hp']
    features = ['value_kwh', 'temp', 'mean_temp', 'min_temp']

    def __init__(self, num_classes: int, sequence_length: int, augmentation_shift: int = None):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.augmentation_shift = self._get_augmentation_shift(augmentation_shift)

    def create_train_test_data(self, path: str, id_col: str, train_split=0.5, balance: str = 'unbalanced',  write: bool = True):
        """
        interface to TSCDataLoader class. reads, transforms, augments and standardizes data.
        :param path: str. path to file
        :param balance: str. sampling strategy for imbalanced data; default is unbalanced
        :param standardize bool. if data should be standardized
        :param write: bool. if train/test files should be written to pickle.
        :return: tuple with train, test data; x: 3D-tensor, y: 1D-array
        """

        train_ids, test_ids = self._get_ids(path, train_split=train_split, id_col=id_col)
        train_arrays, test_arrays = self._read_file(path, train_ids, test_ids)
        x_train, y_train = self._prepare_data(train_arrays,
                                              augment_shift=self.augmentation_shift,
                                              balance=balance)
        x_test, y_test = self._prepare_data(test_arrays, balance=balance)
        if write:
            _path = '../data/model_input/ckw/detection'
            self._write_file((x_train, y_train), _path, 'train')
            self._write_file((x_test, y_test), _path, 'test')
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

    def _prepare_data(self, arrays, augment_shift: int = None, balance: str = 'unbalanced'):
        """
        splits input into equal length sequences. uses window slicing to augment data
        :param arrays: list of np.ndarrays
        :param balance: str. can be eiter over-sampling, under-sampling or unbalanced
        :return: tuple X, y
        """
        assert balance in self._sampling_methods
        xy = []
        for arr in arrays:
            arr = augment_seq(arr, self.sequence_length+1, self._get_augmentation_shift(augment_shift))
            arr = differentiate(arr, dim=[0])
            arr = remove_na(arr)
            if arr.shape[0] == 0:
                continue
            xy.append(arr)

        xy = np.concatenate(xy, axis=0)
        y = np.min(xy[:, :, -1], axis=1)
        X = xy[:, :, :-1]

        if balance == 'unbalanced':
            return X, y
        x_shape = np.array(X.shape)
        if balance == 'over-sampling':
            sampler = RandomOverSampler()
        elif balance == 'under-sampling':
            sampler = RandomUnderSampler()
        X, y = sampler.fit_resample(X.reshape([X.shape[0], -1]), y)
        x_shape[0] = X.shape[0]
        return unison_shuffled_copies(X.reshape(x_shape), y)

    def _write_file(self, data: tuple, path: str, split: str, mean: float = None, s: float = None):
        fname = f'{path}/ts_{self.sequence_length}_{self.augmentation_shift}_{split}'
        with open(f'{fname}.p', 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def _get_ids(path: str, train_split: float = 0.8, id_col='VID') -> tuple:
        """
        split ids into train and test, keeping ratio
        :return: tuple train test ids
        """
        df_ids = pd.read_pickle(path + IDS_FILENAME)
        train_ids = df_ids.groupby('hp').sample(frac=train_split, random_state=1)
        test_ids = df_ids[~df_ids[id_col].isin(train_ids[id_col])]
        #test_ids = test_ids.groupby('hp').sample(n=500)
        # ToDo: assert ratios are as in base
        print('LEN TEST IDS = ', len(test_ids))
        return list(train_ids[id_col]), list(test_ids[id_col])

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


if __name__ == '__main__': # ckw arr shape : (3, 1344, 4)
    dataloader = TSCDataLoader(num_classes=2, sequence_length=672, augmentation_shift=672)
    path = '../data/clean/ckw/detection/'

    for i in [0.8]:
        print(i)
        data = dataloader.create_train_test_data(
            path=path,
            id_col='id',
            train_split=i,
            balance='under-sampling'
        )
        x_train, x_test, y_train, y_test = data
        print(x_train.shape, y_train.shape, y_train.sum())
