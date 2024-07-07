import numpy as np
from sklearn.linear_model import LinearRegression


def get_correlations(df):
    df = df[~df.index.duplicated(keep='first')]
    smd = np.array(df.value_kwh).astype(float)
    weather = np.array(df.temp).astype(float)

    dm30 = np.sum(smd.reshape(-1, 2), axis=1).reshape((7, -1)).T
    dm30[np.isinf(dm30)] = np.nan

    dw30 = np.array(weather)[::2].reshape((7, -1)).T
    dw30[np.isinf(dw30)] = np.nan

    D = {'cor_overall': np.corrcoef(dm30.flatten(), dw30.flatten(), rowvar=False)[0, 1]}

    M = LinearRegression().fit(np.nanmean(dm30, axis=0).reshape(-1, 1), np.nanmean(dw30, axis=0).reshape(-1, 1))
    D['cor_daily'] = M.coef_[0][0]

    for i, (start, end) in enumerate([(0, 12), (12, 36), (38, 44)]):
        M = LinearRegression().fit(dm30[start:end, :5].flatten().reshape(-1, 1),
                                   dw30[start:end, :5].flatten().reshape(-1, 1))
        D[f"cor_{['night', 'daytime', 'evening'][i]}"] = M.coef_[0][0]

        cmin = np.nanmin(dm30, axis=0)
        wmin = np.nanmin(dw30, axis=0)
        M = LinearRegression().fit(cmin.reshape(-1, 1), wmin.reshape(-1, 1))
        D['cor_minima'] = M.coef_[0][0]

        cmax = np.nanmax(dm30, axis=0)
        M = LinearRegression().fit(cmax.reshape(-1, 1), wmin.reshape(-1, 1))
        D['cor_maxmin'] = M.coef_[0][0]

        c_wd = np.nanmean(dm30[:, :5])
        t_wd = np.nanmean(dw30[:, :5])
        c_we = np.nanmean(dm30[:, 5:], axis=1)
        t_we = np.nanmean(dw30[:, 5:], axis=1)
        D['cor_weekday_weekend'] = np.nanmean((c_wd - c_we) / (t_wd - t_we))

    return D


if __name__ == '__main__':
    pass
