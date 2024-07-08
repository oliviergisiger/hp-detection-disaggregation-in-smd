import pickle

from models.models_hp_detection.baseline.baseline_model import BaselineModel


if __name__ == '__main__':

    with open('../data/model_input/ckw/detection/ts_672_672_train.p', 'rb') as f:
        X_train, y_train = pickle.load(f)

    print(
        f'training model on data with shapes:\n'
        f'X_train: {X_train.shape}\ny_train: {y_train.shape}\n'
        f'Class dirstribution in train data: '
        f'\n\thp = 1: {y_train.sum()}\n\thp = 0: {y_train.shape[0] - y_train.sum()}'
    )

    classifier = BaselineModel()
    classifier.train(X_train, y_train)