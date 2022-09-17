import io
import os
from typing import List
import requests
import datetime
import pandas as pd
from dataclasses import dataclass

INMET_GEOGRAFICO_DIR = "/home/b-rbmp-ideapad/Documents/GitHub/ml-solar-forecaster/dados/inmet"
NASA_GEOGRAFICO_DIR = "/home/b-rbmp-ideapad/Documents/GitHub/ml-solar-forecaster/dados/nasa/"

@dataclass
class NasaDownloadEstacaoData:
    """Classe para guardar informações passadas para requisição do Power Nasa"""

    longitude: float
    latitude: float
    altitude: float
    start_date: datetime.date
    end_date: datetime.date
    codigo_estacao_inmet: str


def download_dados_nasa(longitude: float, latitude: float, altitude: float, start_date: datetime.date, end_date: datetime.date, codigo_estacao_inmet: str):
    start_date = end_date - datetime.timedelta(days=365*10)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    url1 = f"https://power.larc.nasa.gov/api/temporal/hourly/point?start={start_date_str}&end={end_date_str}&latitude={latitude}&longitude={longitude}&community=ag&parameters=T2M,T2MDEW,T2MWET,QV2M,RH2M&format=csv&user=brbmp0&header=false&time-standard=utc&site-elevation={altitude:.2f}"
    r1 = requests.get(url=url1)
    url2 = f"https://power.larc.nasa.gov/api/temporal/hourly/point?start={start_date_str}&end={end_date_str}&latitude={latitude}&longitude={longitude}&community=ag&parameters=PRECTOTCORR,WD10M,WS10M,WS50M,WD50M&format=csv&user=brbmp0&header=false&time-standard=utc&site-elevation={altitude:.2f}"
    r2 = requests.get(url=url2)

    df_1 = pd.read_csv(io.StringIO(r1.content.decode('utf-8')), sep=",")
    df_2 = pd.read_csv(io.StringIO(r2.content.decode('utf-8')), sep=",")
    df_2.drop(["YEAR", "MO", "DY", "HR"], axis=1, inplace=True)
    df_concatenado = pd.concat([df_1, df_2], axis=1)

    df_concatenado.to_csv(path_or_buf=NASA_GEOGRAFICO_DIR+codigo_estacao_inmet+".csv")

def gerar_lista_de_downloads() -> List[NasaDownloadEstacaoData]:
    lista_download: List[NasaDownloadEstacaoData] = []
    filenames = [name for name in os.listdir(INMET_GEOGRAFICO_DIR) if os.path.isfile(os.path.join(INMET_GEOGRAFICO_DIR, name))]
    for f_name in filenames:
        file = os.path.join(INMET_GEOGRAFICO_DIR, f_name)  # full path
        df_inmet = pd.read_csv(
            filepath_or_buffer=file, sep=";", header=9
        )
        # Separa as datas
        df_inmet = df_inmet["Data Medicao"]
        data_inicial_str = df_inmet.iloc[0]
        data_final_str = df_inmet.iloc[-1]
        data_inicial = datetime.date(year=int(data_inicial_str[0:4]), month=int(data_inicial_str[5:7]),  day=int(data_inicial_str[8:10]))
        data_final = datetime.date(year=int(data_final_str[0:4]), month=int(data_final_str[5:7]),  day=int(data_final_str[8:10]))


        df_header = pd.read_csv(
            filepath_or_buffer=file, sep=":"
        ).head(7)

        latitude = float(df_header.iloc[1][1])
        longitude = float(df_header.iloc[2][1])
        altitude = float(df_header.iloc[3][1])
        codigo_estacao_inmet = f_name.split("_")[1]

        download_item = NasaDownloadEstacaoData(longitude=longitude, latitude=latitude, altitude=altitude, start_date=data_inicial, end_date=data_final, codigo_estacao_inmet=codigo_estacao_inmet)
        lista_download.append(download_item)



    return lista_download



lista_de_downloads = gerar_lista_de_downloads()
for download_item in lista_de_downloads:
    download_dados_nasa(longitude=download_item.longitude, latitude=download_item.latitude, altitude=download_item.altitude, start_date=download_item.start_date, end_date=download_item.end_date, codigo_estacao_inmet=download_item.codigo_estacao_inmet)
#/api/temporal/hourly/point?parameters=WS10M,WD10M,T2MDEW,T2MWET,T2M,V10M,RH2M,PS,PRECTOT,QV2M,U10M&community=SB&longitude=0&latitude=0&start=20170101&end=20170102&format=CSV
