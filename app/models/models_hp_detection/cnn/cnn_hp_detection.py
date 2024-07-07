import pickle

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import Model
import keras
from datetime import datetime
import numpy as np


class TSCConvNet:

    def __init__(self, input_shape: tuple, num_classes: int, pooling: int,
                 class_weights: [] = None, name: str = None):
        self.input_shape = input_shape[1:]
        self.num_classes = num_classes
        self.pooling = pooling
        self.model = self.__make_model(name)
        self.x_scaler = None
        self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.loss = keras.losses.BinaryCrossentropy()
        self.class_weights = self.get_class_weights(class_weights)

    def _get_callbacks(self, **kwargs):
        model_name = f'{self.model.name}_best_binary_acc.h5'
        return [
            keras.callbacks.ModelCheckpoint(model_name, save_best_only=True, monitor="val_binary_accuracy"),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.0001, patience=10)
        ]

    def __make_model(self, name: str):
        input_layer = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv1D(filters=20, kernel_size=3, padding='same', name='conv_1')(input_layer)
        conv1 = keras.layers.BatchNormalization(name='batch_norm_1')(conv1)
        conv1 = keras.layers.MaxPool1D(self.pooling, name='max_pool_1')(conv1)
        conv1 = keras.layers.ReLU(name='relu_1')(conv1)
        conv1 = keras.layers.Dropout(0.1)(conv1)

        conv2 = keras.layers.Conv1D(filters=20, kernel_size=5, padding='same', name='conv_2')(conv1)
        conv2 = keras.layers.BatchNormalization(name='batch_norm_2')(conv2)
        conv2 = keras.layers.MaxPool1D(self.pooling)(conv2)
        conv2 = keras.layers.ReLU()(conv2)
        conv2 = keras.layers.Dropout(0.1)(conv2)

        conv3 = keras.layers.Conv1D(filters=30, kernel_size=7, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization(name='batch_norm_3')(conv3)
        conv3 = keras.layers.MaxPool1D(self.pooling)(conv3)
        conv3 = keras.layers.ReLU()(conv3)
        conv3 = keras.layers.Dropout(0.1)(conv3)

        flatten = keras.layers.Flatten()(conv3)
        fc1 = keras.layers.Dense(32)(flatten)
        fc1 = keras.layers.Dropout(0.5)(fc1)

        output_layer = keras.layers.Dense(self.num_classes-1, activation='sigmoid')(fc1)


        kwargs = self._create_model_name(name)
        return Model(inputs=input_layer, outputs=output_layer, **kwargs)

    def train(self, epochs: int, X: np.ndarray, y: np.ndarray, save: bool = True, **kwargs):
        self.get_x_scaler(X=X)
        scaled_X = self.scale_x(X)

        self.model.compile(self.optimizer,
                           loss=self.loss,
                           metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))

        self.model.fit(scaled_X, y, batch_size=32,
                       epochs=epochs,
                       shuffle=True,
                       validation_split=0.5,
                       callbacks=self._get_callbacks())
        if save:
            model_file = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{self.model.name}.h5'
            scaler_file = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{self.model.name}_scaler.p'
            self.model.save(model_file)
            pickle.dump(self.x_scaler, open(scaler_file, 'wb'))

    def predict(self, X: np.ndarray, binary=True, thr: float = 0.5):
        pred = self.model.predict(self.scale_x(X))
        if binary:
            return np.where(pred >= thr, 1, 0)
        return pred

    def load_model(self, path_to_model: str):
        scaler_file = path_to_model[:-3] + '_scaler.p'
        self.model = keras.models.load_model(path_to_model, compile=False)
        self.x_scaler = pickle.load(open(scaler_file, 'rb'))

    def get_x_scaler(self, X: np.ndarray):
        scaler = StandardScaler()
        self.x_scaler = scaler.fit(X.reshape(-1, X.shape[-1]))

    def scale_x(self, X: np.ndarray):
        X_scaled_reshaped = self.x_scaler.transform(X.reshape(-1, X.shape[-1]))
        return X_scaled_reshaped.reshape(X.shape)

    @staticmethod
    def weighted_cross_entropy(logits, labels, class_weights):
        return tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=class_weights)

    @staticmethod
    def _create_model_name(name: str):
        if name:
            return dict(name=name)
        return dict()

    @staticmethod
    def get_class_weights(weights):
        if weights:
            return tf.constant(weights, dtype=tf.float32)
        return tf.constant([1.0, 1.0], dtype=tf.float32)



if __name__ == '__main__':
    classifier = TSCConvNet(input_shape=(10, 1344, 5),
                            pooling=2,
                            num_classes=2,
                            class_weights=[1.0, 2.0])
    print(classifier.model.summary())
