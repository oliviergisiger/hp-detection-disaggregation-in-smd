import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

SAVE_MODEL_PATH = 'models/models_hp_detection/baseline/trained_models'

def eucledean_error(predictions, truths):
    return np.linalg.norm(predictions-truths)


def metrics(pred, truth):
    n = truth.shape[0]

    tp = ((pred == 1) & (truth == 1)).sum()
    fp = ((pred == 1) & (truth == 0)).sum()
    fn = ((pred == 0) & (truth == 1)).sum()

    accuracy = np.round(((truth == pred).sum() / n), 3)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


class BaselineModel:
    model_path = SAVE_MODEL_PATH

    @staticmethod
    def model(x, p, s, use_gradient=True):
        """
        x: np.array, conatining sequences of equal length
        p: threshold gradient
        s: threshold of how many p in to assign heatpump label to sequence
        use_gradient: if True uses gradient, else usese relative change
        """
        pred = []
        for i in x:
            _i = np.roll(i, 1)
            _x = i / _i
            if use_gradient:
                _x = i - _i
            _x = (_x > p).astype(int)

            _y = (_x.sum() > s).astype(int)
            pred.append(_y)

        return np.array(pred)

    def train(self, x, y, use_gradients=True, target_metric='f1', save=True):
        _metrics = ('acc', 'precision', 'recall', 'f1')
        assert target_metric in _metrics, f'target metric not in {_metrics}!'

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2
        )

        params = {}
        val_params = {}

        for s in range(int(np.round(x.shape[1] / 4))):
            for p in np.arange(0.1, 6, 0.1):
                y_pred = self.model(x_train, p, s, use_gradient=use_gradients)
                euc_error = eucledean_error(y_pred, y_train)
                acc, prec, rec, f1 = metrics(y_pred, y_train)

                val_euc_error = eucledean_error(y_pred, y_train)
                val_acc, val_prec, val_rec, val_f1 = metrics(y_pred, y_train)

                params[(p, s)] = euc_error, acc, prec, rec, f1
                val_params[(p, s)] = val_euc_error, val_acc, val_prec, val_rec, val_f1

        if save:
            acc_dict = {k: v[1] for k, v in val_params.items()}
            error_dict = {k: v[0] for k, v in val_params.items()}
            prec_dict = {k: v[2] for k, v in val_params.items()}
            rec_dict = {k: v[3] for k, v in val_params.items()}
            f1_dict = {k: v[4] for k, v in val_params.items()}

            lp, ls = zip(*acc_dict.keys())

            results_df = pd.DataFrame.from_dict(
                {
                    'p': lp,
                    's': ls,
                    'e': error_dict.values(),
                    'acc': acc_dict.values(),
                    'precision': prec_dict.values(),
                    'recall': rec_dict.values(),
                    'f1': f1_dict.values()
                }
            )

            results_df.to_pickle(f'{self.model_path}/baseline_model.p')
            best_val = results_df[target_metric] == max(results_df[target_metric])
            print(results_df.loc[best_val, :])


if __name__ == '__main__':
    ...







