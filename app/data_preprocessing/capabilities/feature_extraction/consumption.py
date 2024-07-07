import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def get_consumption(B):
    B = B[~B.index.duplicated(keep='first')]
    B = np.array(B).astype(float)
    if B.shape[0] < 672:
        pass
    dm15 = B.reshape((7, -1)).T
    dm15_1d = dm15.T.reshape(1, -1)[0]
    D = {}

    D['c15_week'] = np.mean(dm15, where=~np.isnan(dm15))

    weekday = np.arange(1, 5 * 4 * 24 + 1)
    weekend = np.arange(5 * 4 * 24 + 1, 672)
    night = np.arange(1 * 4 + 1, 6 * 4 + 1)
    morning = np.arange(6 * 4 + 1, 10 * 4 + 1)
    noon = np.arange(10 * 4 + 1, 14 * 4 + 1)
    afternoon = np.arange(14 * 4 + 1, 18 * 4 + 1)
    evening = np.arange(18 * 4 + 1, 22 * 4 + 1)

    D['s15_min'] = np.nanmin(dm15)
    D['s15_max'] = np.nanmax(dm15)
    D['r15_mean_max'] = D['c15_week'] / D['s15_max']
    D['r15_min_mean'] = D['s15_min'] / D['c15_week']
    D['s15_we_max'] = np.nanmax(dm15_1d[weekend])
    D['s15_we_min'] = np.nanmin(dm15_1d[weekend])
    D['s15_wd_max'] = np.nanmax(dm15_1d[weekday])
    D['s15_wd_min'] = np.nanmin(dm15_1d[weekday])
    D['r15_min_wd_we'] = D['s15_wd_min'] / D['s15_we_min'] if D['s15_we_min'] > 0 else 1
    D['r15_max_wd_we'] = D['s15_wd_max'] / D['s15_we_max']

    q = np.nanquantile(dm15, [0.25, 0.5, 0.75])
    D['s15_q1'] = q[0]
    D['s15_q2'] = q[1]
    D['s15_q3'] = q[2]
    D['s15_min_avg'] = np.mean(np.nanmin(dm15, axis=0))
    D['s15_max_avg'] = np.mean(np.nanmax(dm15, axis=0))

    D['s15_variance'] = np.nanvar(dm15_1d[0:672])
    D['s15_var_we'] = np.nanvar(dm15_1d[weekend])
    D['s15_var_wd'] = np.nanvar(dm15_1d[weekday])
    D['r15_var_wd_we'] = D['s15_var_wd'] / D['s15_var_we']

    D['s15_cor'] = np.nanmean(np.corrcoef(dm15, rowvar=False))
    D['s15_cor_we'] = np.nanmean(np.corrcoef(dm15[:, 5:7], rowvar=False))
    D['s15_cor_wd'] = np.nanmean(np.corrcoef(dm15[:, 0:5], rowvar=False))

    profile_wd = np.mean(dm15[:, 0:5], axis=1)
    profile_we = np.mean(dm15[:, 5:7], axis=1)
    D['s15_cor_wd_we'] = pearsonr(profile_wd, profile_we)[0]

    D['s15_sm_variety'] = np.nanquantile(np.abs(np.diff(B)), 0.2)
    D['s15_bg_variety'] = np.nanquantile(np.abs(np.diff(B)), 0.6)
    # ToDo
    # D['s15_sm_max'] = 0
    D['s15_sm_max'] = np.mean([np.max(0.5 * V[1:-1] + 0.25 * (V[2:] + V[:-2])) for V in dm15[:, 0:5].T])
    D['s15_number_zeros'] = np.sum(B == 0)

    daily = np.sum(dm15[evening - 1, 0:5] - D['s15_min'])
    D['c15_evening_no_min'] = np.mean(daily)
    daily = np.sum(dm15[morning - 1, 0:5] - D['s15_min'])
    D['c15_morning_no_min'] = np.mean(daily)
    daily = np.sum(dm15[night - 1, 0:5] - D['s15_min'])
    D['c15_night_no_min'] = np.mean(daily)
    daily = np.sum(dm15[noon - 1, 0:5] - D['s15_min'])
    D['c15_noon_no_min'] = np.mean(daily)
    daily = np.sum(dm15[afternoon - 1, 0:5] - D['s15_min'])
    D['c15_afternoon_no_min'] = np.mean(daily)


    _v = [((np.max(V) - np.min(V)) / (np.mean(V) - np.min(V))) for V in dm15[:, 0:5].T]
    D['r15_mean_max_no_min'] = min(10, np.mean(_v))

    _v = [(np.sum(V[evening - 1] - np.min(V)) / np.sum(V[20:28] - np.min(V))) for V in dm15[:, 0:5].T]
    D['r15_evening_noon_no_min'] = min(10, np.mean(_v))

    _v = [(np.sum(V[morning - 1] - np.min(V)) / np.sum(V[20:28] - np.min(V))) for V in dm15[:, 0:5].T]
    D['r15_morning_noon_no_min'] = min(10, np.mean(_v))

    _v = [(np.sum(V[night - 1] - np.min(V)) / np.sum(V[2:10] - np.min(V))) for V in dm15[:, 0:5].T]
    D['r15_day_night_no_min'] = min(10, np.mean(_v))

    D['t15_above_0.5kw'] = np.sum(dm15 > 0.5)
    D['t15_above_1kw'] = np.sum(dm15 > 1)
    D['t15_above_2kw'] = np.sum(dm15 > 2)
    D['t15_above_mean'] = np.sum(dm15 > D['c15_week'])
    D['t15_daily_max'] = np.argmax(dm15)
    D['t15_daily_min'] = np.argmin(dm15)

    return D


if __name__ == '__main__':
    df = pd.read_pickle('data/clean/hopf/data.p')
    print(df.head())

