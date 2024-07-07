import pandas as pd
import requests
from domain import secrets

from data_preprocessing.ports.base_classes import APISource


class MeteoStatSource(APISource):

    def __init__(self, url: str):
        self._url = url


    def get_data(self, station: str, start: str, end: str):
        query = {"station": station,
                 "start": start,
                 "end": end}

        headers = {
            "X-RapidAPI-Key": secrets.METEOSTAT.API_KEY,
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }

        response = requests.get(self._url,
                                headers=headers,
                                params=query)

        return response.json()


if __name__ == '__main__':
    url = 'https://meteostat.p.rapidapi.com/stations/hourly'
    source = MeteoStatSource(url)
    start, end = '2022-10-31', '2023-02-28'
    data = source.get_data('06650', start, end)
    df = pd.DataFrame.from_dict(data['data'])
    df.to_pickle('data/raw/ckw/weather.p')




