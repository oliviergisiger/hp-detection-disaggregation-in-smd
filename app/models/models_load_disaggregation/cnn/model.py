import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
import keras
from datetime import datetime
import numpy as np


class DisaggregationConvNet:

    def __init__(self, input_shape: tuple, num_aux_features: int, pooling: int,
                 dilation: int = 1, name: str = None, model_path: str = None):
        self.input_shape = input_shape[1:]
        self.output_size = input_shape[1]
        self.num_aux_features = num_aux_features
        self.pooling = pooling
        self.dilation = dilation
        self.model = self.__make_model(name)
        self.x_scaler = None
        self.aux_scaler = None
        self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.loss = keras.losses.MeanSquaredError()
        self.model_path = model_path

    def _get_callbacks(self, **kwargs):
        model_name = f'{self.model_path}/{self.model.name}_best_mse.h5'
        return [
            keras.callbacks.ModelCheckpoint(model_name, save_best_only=True, monitor="val_mean_squared_error"),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.0001, patience=10),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        ]

    def __make_model(self, name: str):
        input_layer = keras.layers.Input(self.input_shape, name='sequence_input')
        aux_input = keras.layers.Input(self.num_aux_features)

        conv1 = keras.layers.Conv1D(filters=32, kernel_size=7, padding='same', name='conv_1')(input_layer)
        conv1 = keras.layers.MaxPooling1D(pool_size=2, name='pool_1')(conv1)
        conv1 = keras.layers.ReLU(name='relu_1')(conv1)

        conv2 = keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', name='conv_2')(conv1)
        conv2 = keras.layers.MaxPooling1D(pool_size=2, name='pool_2')(conv2)
        conv2 = keras.layers.ReLU(name='relu_2')(conv2)

        conv3 = keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', name='conv_4')(conv2)
        conv3 = keras.layers.MaxPooling1D(pool_size=2, name='pool_4')(conv3)
        conv3 = keras.layers.ReLU(name='relu_4')(conv3)

        flatten = keras.layers.Flatten()(conv3)

        concatenate = keras.layers.Concatenate()([flatten, aux_input])
        dense1 = keras.layers.Dense(self.input_shape[0]*2, activation='relu')(concatenate)
        output_layer = keras.layers.Dense(self.output_size, activation='relu')(dense1)

        kwargs = self._create_model_name(name)
        return Model(inputs=[input_layer, aux_input], outputs=output_layer, **kwargs)

    def train(self, epochs: int, X: np.ndarray, aux: np.ndarray, y: np.ndarray, save: bool = True, **kwargs):
        print(f'start training model:\ncallbacks: {self._get_callbacks()}\nepochs: {epochs}')
        self.get_x_scaler(X=X)
        scaled_X = self.scale_x(X)
        self.get_aux_scaler(aux=aux)
        scaled_aux = self.scale_aux(aux)

        self.model.compile(self.optimizer, loss=self.loss,
                           metrics=[tf.keras.metrics.MeanSquaredError(),
                                    tf.keras.metrics.RootMeanSquaredError()])
        if save:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_file = f'{self.model_path}/{timestamp}_{self.model.name}.h5'
            scaler_file = f'{self.model_path}/{timestamp}_{self.model.name}_scaler.p'
            self.model.save(model_file)
            pickle.dump([self.x_scaler, self.aux_scaler], open(scaler_file, 'wb'))

    def predict(self, X: np.ndarray, aux: np.ndarray):
        return self.model.predict([self.scale_x(X), self.scale_aux(aux)])

    def load_model(self, path_to_model: str):
        scaler_file = path_to_model[:-3] + '_scaler.p'
        self.model = keras.models.load_model(path_to_model, compile=False)
        self.x_scaler, self.aux_scaler = pickle.load(open(scaler_file, 'rb'))

    def get_x_scaler(self, X: np.ndarray):
        scaler = MinMaxScaler()
        self.x_scaler = scaler.fit(X.reshape(-1, X.shape[-1]))
        print('set X sclaer')

    def scale_x(self, X: np.ndarray):
        X_scaled_reshaped = self.x_scaler.transform(X.reshape(-1, X.shape[-1]))
        return X_scaled_reshaped.reshape(X.shape)

    def get_aux_scaler(self, aux: np.ndarray):
        scaler = MinMaxScaler()
        self.aux_scaler = scaler.fit(aux)
        print(f'set aux_scaler')

    def scale_aux(self, aux: np.ndarray):
        return self.aux_scaler.transform(aux)

    @staticmethod
    def _create_model_name(name: str):
        if name:
            return dict(name=name)
        return dict()


if __name__ == '__main__':
    disaggregator = DisaggregationConvNet(input_shape=(100, 192, 7),
                                          name='dummy',
                                          num_aux_features=1,
                                          pooling=2)
    print(disaggregator.model.summary())