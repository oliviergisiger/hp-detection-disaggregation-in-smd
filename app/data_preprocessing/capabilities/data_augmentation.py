import numpy as np


def augment_seq(seq, seq_size, shift):
    """
    returns numpy array of size (len(seq)-seq_size, seq_size)
    3d tensor-slicing, let x be a 3d tensor:
        - x[:, 0, 0] -> all sequences, first observation, first predictor
        - x[0, :, 0] -> first sequence, all observations, first predictor
        - x[0, 0, :] -> first sequence, first observation, all predictors
    """
    l = [np.array(seq[s:s+seq_size, :]) for s in range(0, (len(seq)-seq_size), shift)]
    return np.array(l)


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


def standardize_2d(x, mode='train', mean=None, s=None):
    if mode == 'train':
        mean = np.nanmean(x, axis=0)
        s = np.nanstd(x, axis=0)
        s = np.where(s == 0, 1, s)
    x_norm = (x - mean) / s
    return x_norm.reshape(x.shape), mean, s


def differentiate(x: np.ndarray):
    return np.diff(x, axis=1)


def remove_na(m, axis=1):
    dim_in = np.array(m.shape)
    m = m.reshape([dim_in[0], -1])
    m = m[~np.isnan(m).any(axis=axis)]
    dim_in[0] = m.shape[0]
    return m.reshape(dim_in)


if __name__ == '__main__':
    pass



