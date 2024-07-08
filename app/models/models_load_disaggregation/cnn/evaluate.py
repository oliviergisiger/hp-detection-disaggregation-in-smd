import pickle
import sklearn.metrics as metrics

from models.models_load_disaggregation.cnn.model import DisaggregationConvNet


LOAD_MODEL_PATH = 'models/models_load_disaggregation/cnn/trained_models/'
MODEL = '20240708-205311_disaggregation_cnn_672.h5'
TEST_FILE = '../data/model_input/ckw/disaggregation/ts_672_672_test.p'


if __name__ == '__main__':

    with open(TEST_FILE, 'rb') as data:
        x_test, x_aux_test, y_test = pickle.load(data)

    disaggregator = DisaggregationConvNet(input_shape=x_test.shape, num_aux_features=4, pooling=1)
    disaggregator.load_model(LOAD_MODEL_PATH + MODEL)
    y_pred = disaggregator.predict(x_test, aux=x_aux_test)

    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    print(f'R2: {r2}\nMAE: {mae}\nRMSE: {rmse}')
