import pickle
from sklearn.metrics import roc_auc_score, classification_report
from models.models_hp_detection.cnn.model import TSCConvNet

LOAD_MODEL_PATH = 'models/models_hp_detection/cnn/trained_models/'
MODEL = '20240708-063451_cnn_detection_672.h5'
TEST_FILE = '../data/model_input/ckw/detection/ts_672_672_test.p'


if __name__ == '__main__':

    with open(TEST_FILE, 'rb') as data:
        x_test, y_test = pickle.load(data)

    classifier = TSCConvNet(input_shape=x_test.shape,
                            num_classes=2,
                            class_weights=[1.0, 1.0],
                            pooling=2)
    classifier.load_model(LOAD_MODEL_PATH + MODEL)
    y_pred = classifier.predict(x_test, binary=True, thr=0.8)

    print(classification_report(y_true=y_test, y_pred=y_pred, digits=3))
    print('Area under the (ROC-) Curve', roc_auc_score(y_test, y_pred))
