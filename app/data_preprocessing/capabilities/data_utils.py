import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d

def label_row(x, a_th, l_th, add_column=False):
    """
    x: np.array, conatining sequences of equal length
    a_th: threshold to sign an observation as heatpump activity
    l_th: threshold of how many observations must be labeled as active
    to assign heatpump label to sequence
    """
    y = ((x > a_th).sum(axis=1) > l_th).astype(int)
    if add_column:
        return np.c_[x, y]
    return y


def unison_shuffled_copies(x, y):
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(len(x))
    return x[p], y[p]


def normalize(x: np.ndarray):
    """
    normalizes x row-wise:
    x: 3D-numpy array
    """
    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    return (x - x_min)/(x_max - x_min)


def normalize_mean_sd(x: np.ndarray, mode='train', mean=None, s=None):
    x_reshaped = x.reshape(-1, 2)
    if mode == 'train':
        mean = np.mean(x_reshaped, axis=0)
        s = np.std(x_reshaped, axis=0)
    x_norm = (x_reshaped - mean) / s
    return x_norm.reshape(x.shape), mean, s


def standardize(x, mode='train', axis=(0, 1), mean=None, s=None):
    """
    standardizes np.ndarray along axis.
    :param x: np.ndarray
    :param mode: train r test. if train, mean and s are calculated. in test-mode, mean and s are expected inputs
    :param axis: for 3d standardization use tuple. for 2d standardization use int. default 3d: tuple
    :param mean: mean of feature
    :param s: sd of feature
    :return: x
    """
    if mode == 'train':
        mean = np.nanmean(x, axis=axis)
        s = np.nanstd(x, axis=axis)
        s = np.where(s == 0, 1, s)
    x_norm = (x - mean) / s
    return x_norm.reshape(x.shape), mean, s


def differentiate(x, dim, n=1):
    if np.isnan(x).all():
        return x
    y = x[:, :, -1].reshape(x.shape[0], x.shape[1], 1)
    diff = np.diff(x[:, :, dim], n=n, axis=1)
    return np.concatenate([x[:, :-n, :-1], diff, y[:, :-n, :]], axis=2)


def get_load_features(array, n=1):
    return np.diff(array, n, append=np.nan)


def convert_time(v: np.ndarray):
    """ converts time to a periodicity, according to bruedermueller"""
    if np.isnan(v).all():
        return v
    s = 1440  # minutes in a day
    x = np.expand_dims(np.sin(2 * np.pi * (v / s)), axis=2)
    return x


def get_temporal_features(array, scale=1440):
    if np.isnan(array).all():
        return array
    return np.sin(2 * np.pi * (array / scale))


def moving_average(array, periods):
    _pad = int(np.floor(periods/2))
    out = np.convolve(array, np.ones(periods)/periods, mode='valid')
    return np.pad(out, _pad, 'constant', constant_values=np.nan)


def moving_min(array, periods):
    return minimum_filter1d(array, periods, mode='constant', cval=-np.inf)


def moving_max(array, periods):
    return maximum_filter1d(array, periods, mode='constant', cval=np.inf)


def get_temperature_features(array, periods=97):
    mean = moving_average(array, periods)
    _min = moving_min(array, periods)
    _max = moving_max(array, periods)
    return np.column_stack([mean, _min, _max])


def get_auxilary_features(x: np.ndarray):
    # load aux:
    x_load = x[:, :, 0]
    _mean = np.mean(x_load, axis=1)
    #_std = np.std(x_load, axis=1)
    _var = np.var(x_load, axis=1)
    _min = np.min(x_load, axis=1)
    _max = np.max(x_load, axis=1)

    return np.column_stack([_min, _max, _mean, _var])



def remove_na(m, axis=1):
    if np.isnan(m).all():
        return m
    dim_in = np.array(m.shape)
    m = m.reshape([dim_in[0], -1])
    m = m[~np.isnan(m).any(axis=axis)]
    dim_in[0] = m.shape[0]
    return m.reshape(dim_in)


if __name__ == '__main__':
    m = np.random.randint(0, 100, [33, 672, 5])
    m_out = differentiate(m, 2)
    print(m_out)
