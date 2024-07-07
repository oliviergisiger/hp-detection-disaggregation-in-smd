import pandas as pd
from typing import Dict

from data_preprocessing.ports.base_classes import FileSource


class CKWFileSource(FileSource):

    def __init__(self):
        pass

    def read_file(self, path) -> pd.DataFrame | Dict:
        return pd.read_pickle(path)

    def read_multiple_files(self, paths) -> pd.DataFrame:
        dfs = [self.read_file(path) for path in paths]
        for _df in dfs:
            _df.timestamp = pd.to_datetime(_df.timestamp)
        full_df = pd.concat(dfs)
        return full_df


    def process_data(self, path: str) -> pd.DataFrame:
        pass


class CKWWeatherSource(FileSource):

    def __init__(self):
        pass

    def read_file(self, path) -> pd.DataFrame | Dict:
        df = pd.read_pickle(path)
        # extract daily features
        daily_features = df.groupby(df.timestamp.dt.date).agg(min_temp=('temp', 'min'),
                                                              mean_temp=('temp', 'mean')).reset_index()
        df = pd.merge(df, daily_features, left_on=df.timestamp.dt.date, right_on='timestamp', suffixes=('', '_y'))

        df = df[['timestamp', 'temp', 'min_temp', 'mean_temp']].set_index('timestamp')
        return df.resample('15min').ffill().reset_index()


    def process_data(self, path: str) -> pd.DataFrame:
        pass



if __name__ == '__main__':
    MONTHS = ['2022-11', '2022-12', '2023-01', '2023-02']

    PATHS_IN_HP = [f'data/raw/ckw/{month}_hp_data_clean.p' for month in MONTHS]
    PATHS_IN_NO_HP = [f'data/raw/ckw/{month}_hp_negative_data_clean.p' for month in MONTHS]
    PATH_WEATHER = 'data/raw/ckw/weather.p'


    source = CKWFileSource()
    df = source.read_multiple_files(PATHS_IN_HP)
    print(df.head())