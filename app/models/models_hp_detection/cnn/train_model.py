import pickle
from models.models_hp_detection.cnn.cnn_hp_detection import TSCConvNet

train_file = 'ts_672_672_train.p'
save_model_path = 'models/models_hp_detection/cnn/trained_models'

if __name__ == '__main__':

    with open(f'../data/model_input/ckw/detection/{train_file}', 'rb') as data:
        x_train, y_train = pickle.load(data)

    print(
        f'training model on data with shapes:\n'
        f'X_train: {x_train.shape}\ny_train: {y_train.shape}\n'
        f'Class dirstribution in train data: '
        f'\n\thp = 1: {y_train.sum()}\n\thp = 0: {y_train.shape[0] - y_train.sum()}'
    )

    classifier = TSCConvNet(input_shape=x_train.shape,
                            num_classes=2,
                            pooling=2,
                            class_weights=[1.0, 1.0],
                            name=f'{save_model_path}/cnn_detection_672',
                            )
    classifier.train(epochs=10, X=x_train, y=y_train, save=False)



