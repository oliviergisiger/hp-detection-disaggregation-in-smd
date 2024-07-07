from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from pathlib import Path


class FileSource(ABC):
    """
    abstract class to serve as lowest level abstraction of interface for
    source files, such as ckw data or general files.
    """

    @abstractmethod
    def read_file(self, path) -> pd.DataFrame | Dict:
        pass

    @abstractmethod
    def process_data(self, path: str) -> pd.DataFrame:
        pass


class FileSink(ABC):
    export_modes = ('single', 'full')

    @abstractmethod
    def write_household_file(self, path: str, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def write_complete_file(self, path: str, df: pd.DataFrame) -> None:
        pass

    @staticmethod
    def mkdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)


class APISource(ABC):

    @abstractmethod
    def get_data(self, **kwargs):
        pass


class DataLoader(ABC):
    ...
