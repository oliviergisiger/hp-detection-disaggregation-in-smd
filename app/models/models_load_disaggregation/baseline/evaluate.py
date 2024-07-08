import numpy as np
import pickle
from sklearn import metrics

from models.models_load_disaggregation.baseline.baseline_model import BaselineDisaggregatior


if __name__ == '__main__':

    with open('../data/model_input/ckw/disaggregation/ts_672_672_test.p', 'rb') as data:
        x_test, x_aux_test, y_test = pickle.load(data)


        y_pred = []

        for obs in range(x_test.shape[0]):
            p = BaselineDisaggregatior(x_test[obs, :]).decompose(plot=False)[:, 1]
            y_pred.append(p)

        y_pred = np.array(y_pred)

        rmse = metrics.root_mean_squared_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        print(f'R2: {r2}\nMAE: {mae}\nRMSE: {rmse}')