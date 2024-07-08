import pickle
from models.models_load_disaggregation.cnn.model import DisaggregationConvNet
import numpy as np

TRAIN_FILE = '../data/model_input/ckw/disaggregation/ts_672_672_train.p'
SAVE_MODEL_PATH = 'models/models_load_disaggregation/cnn/trained_models'


if __name__ == '__main__':
    with open(TRAIN_FILE, 'rb') as data:
        x_train, aux_train, y_train = pickle.load(data)

    print(
        f'training model on data with shapes:\n'
        f'X_train: {x_train.shape}\nauxilary_train: {aux_train.shape}\ny_train: {y_train.shape}\n'
    )

    disaggregator = DisaggregationConvNet(
        input_shape=x_train.shape,
        pooling=1,
        num_aux_features=4,
        name='disaggregation_cnn_672',
        model_path=SAVE_MODEL_PATH
    )
    disaggregator.train(epochs=20, X=x_train, aux=aux_train, y=y_train, save=True)