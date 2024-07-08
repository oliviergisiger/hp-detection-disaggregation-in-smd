import pandas as pd
import numpy as np

from domain import configs

from data_preprocessing.adapter.ckw_source import CKWWeatherSource, CKWFileSource
from data_preprocessing.adapter.ckw_sink import CKWFileSink

MONTHS = configs.DATA.MONTHS
PATH_WEATHER = configs.DATA.PATHS.WEATHER
NILM_PATHS_IN_HP = [f'../data/raw/ckw/{month}_hp_disagg_data.p' for month in MONTHS]
NILM_PATHS_IN_MASKED = [f'../data/raw/ckw/{month}_hp_data.p' for month in MONTHS]


class PreProcessCKWDataDisAgg:

    def __init__(self, file_source: CKWFileSource, weather_source: CKWWeatherSource, sink: CKWFileSink):
        self._file_source = file_source
        self._weather_source = weather_source
        self._sink = sink

    def invoke(self):
        df_weather = self._weather_source.read_file(PATH_WEATHER).set_index('timestamp')
        df_hp_load = self._file_source.read_multiple_files(NILM_PATHS_IN_HP)
        df_masked_load = self._file_source.read_multiple_files(NILM_PATHS_IN_MASKED)

        df = pd.merge(df_hp_load,
                      df_masked_load,
                      on=['id', 'timestamp'],
                      suffixes=('', '_masked')).set_index('timestamp')

        df = pd.merge(df, df_weather, how='left', left_index=True, right_index=True)
        df['dow'] = df.index.dayofweek
        df.loc[:, 'min_of_day'] = df.index.minute + df.index.hour * 60
        df.set_index('id', inplace=True, drop=True)

        self._sink.write_complete_file('../data/clean/ckw/disaggregation', df)
        self._sink.write_meta_file('../data/clean/ckw/disaggregation', pd.Series(df.reset_index()['id'].unique()))


if __name__ == '__main__':
    usecase = PreProcessCKWDataDisAgg(
        file_source=CKWFileSource(),
        weather_source=CKWWeatherSource(),
        sink=CKWFileSink()
    )
    usecase.invoke()
