import pandas as pd

from data_preprocessing.ports.base_classes import FileSink


class CKWFileSink(FileSink):

    def write_household_file(self, path: str, df: pd.DataFrame) -> None:
        pass

    def write_complete_file(self, path: str, df: pd.DataFrame) -> None:
        self.mkdir(path)
        df.to_pickle(f'{path}/data.p')

    def write_meta_file(self, path: str, df: pd.DataFrame):
        self.mkdir(path)
        df.to_pickle(f'{path}/ids.p')
