import pandas as pd
import numpy as np

from domain import configs

from data_preprocessing.adapter.ckw_source import CKWWeatherSource, CKWFileSource
from data_preprocessing.adapter.ckw_sink import CKWFileSink

MONTHS = configs.DATA.MONTHS
PATH_WEATHER = configs.DATA.PATHS.WEATHER

# CLASSIFICATION
PATHS_IN_HP = [f'data/raw/ckw/{month}_hp_data_new_clean.p' for month in MONTHS]
PATHS_IN_NO_HP = [f'data/raw/ckw/{month}_hp_negative_data_new_clean.p' for month in MONTHS]




class PreProcessCKWData:

    def __init__(self, file_source: CKWFileSource, weather_source: CKWWeatherSource, sink: CKWFileSink):
        self._file_source = file_source
        self._weather_source = weather_source
        self._sink = sink

    def invoke(self):
        df_weather = self._weather_source.read_file(PATH_WEATHER)
        df_hp = self._file_source.read_multiple_files(PATHS_IN_HP)
        df_no_hp = self._file_source.read_multiple_files(PATHS_IN_NO_HP)
        ids_hp = df_hp.id.unique()
        ids_no_hp = df_no_hp.id.unique()
        ids_hp = pd.DataFrame(dict(id=ids_hp, hp=np.ones(ids_hp.shape[0])))
        ids_no_hp = pd.DataFrame(dict(id=ids_no_hp, hp=np.zeros(ids_no_hp.shape[0])))
        ids = pd.concat([ids_hp, ids_no_hp])
        df_hp['hp'] = 1
        df_no_hp['hp'] = 0
        df_hp = pd.merge(df_hp, df_weather, how='left')
        df_no_hp = pd.merge(df_no_hp, df_weather, how='left')
        df_full = pd.concat([df_hp, df_no_hp])
        df_full['dow'] = df_full.timestamp.dt.dayofweek
        df_full.set_index('id', inplace=True)

        self._sink.write_complete_file('data/clean/ckw/test_data', df_full)
        self._sink.write_meta_file('data/clean/ckw/test_data', ids)

    def _combine_load_data(self, paths):
        pass

if __name__ == '__main__':
    usecase = PreProcessCKWData(
        file_source=CKWFileSource(),
        weather_source=CKWWeatherSource(),
        sink=CKWFileSink()
    )
    usecase.invoke()
