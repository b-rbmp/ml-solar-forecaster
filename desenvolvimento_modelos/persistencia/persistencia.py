# Imports
from hashlib import sha256
from pandas.util import hash_pandas_object
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import logging
import seaborn as sns
import matplotlib.dates as mdates
import scipy as sp

BASE_DIR = "/home/b-rbmp-ideapad/Documents/GitHub/ml-solar-forecaster/"

class HashableDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)
        
 # Configuracao Logging
def create_logger(debug_mode: bool):
    logging_level: int
    handler: logging.FileHandler
    if not debug_mode:
        handler = logging.FileHandler('persistencia.log')
        logging_level = logging.INFO
    else:
        handler = logging.FileHandler('persistencia.debug')
        logging_level = logging.DEBUG

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger('PersistenciaNasaInmetLogger')
    logger.setLevel(logging_level)
    logger.addHandler(handler)

    return logger


class PersistenciaModel:
    """docstring for PersistenciaModel."""

    def __init__(
        self,
    ):
        self.mae_validacao, self.mae_teste, self.series_validacao, self.series_teste = self.rodar_instancia_treinamento()

    # Scatter plot de Previsões x Target - conjunto de validação
    def plot_scatter_previsoes_targets_validacao(self):
        plt.figure()
        sns.scatterplot(
            x=self.series_validacao["previsao_persistencia"],
            y=self.series_validacao["IRRADIÂNCIA"],
        )
        plt.xlabel("Persistência (W/m²)")
        plt.ylabel("Alvo Validação (W/m²)")
        plt.show()

    # Scatter plot de Previsões x Target - conjunto de teste
    def plot_scatter_previsoes_targets_teste(self):
        plt.figure()
        sns.scatterplot(
            x=self.series_teste["previsao_persistencia"],
            y=self.series_teste["IRRADIÂNCIA"],
        )
        plt.xlabel("Persistência (W/m²)")
        plt.ylabel("Alvo Teste (W/m²)")
        plt.show()

    # Plota as previsões vs target de um dos conjuntos
    def plot_line_previsoes_targets(self, series: pd.DataFrame, numero_amostras: int):
        series_filtered = series.tail(numero_amostras)
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.lineplot(y=series_filtered["IRRADIÂNCIA"], x=pd.to_datetime(series_filtered["Data_Horario"]), ax=ax, label="Alvo")
        sns.lineplot(y=series_filtered["previsao_persistencia"], x=pd.to_datetime(series_filtered["Data_Horario"]), ax=ax, label="Persistência", linestyle='--')

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        date_form = mdates.DateFormatter('%d/%m')
        ax.xaxis.set_major_formatter(date_form)
        plt.legend(loc="upper right")
        plt.ylabel("Irradiância (W/m²)")
        plt.xlabel("Data")
        plt.show()


    # Função que roda o modelo da persistência
    def rodar_instancia_treinamento(self) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(f"{BASE_DIR}dados/pre_processado/salvador.csv")
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

        # Adição de Coluna para Radiação t-1
        # Separacao das Features que vão até t (todas exceto radiação) e features que vão até t-1 para entrada.
        # Para isso será criado uma nova coluna de radiação que é a radiação atrasada de 1h
        # nome_coluna_radiacao_shifted = "RADIACAO t-1"
        # df[nome_coluna_radiacao_shifted] = df["IRRADIÂNCIA"].shift(1)
        # df.fillna(method="bfill", inplace=True)

        # Dropa colunas não utilizadas
        df.drop(["Unnamed: 0"], axis=1, inplace=True)

        # Preparação dos Dados

        # Train-Validation-Test split
        numero_amostras_treinamento = int(0.6 * df.shape[0])  # 60%
        numero_amostras_validacao = int(0.2 * df.shape[0])  # 20%
        numero_amostras_teste = (
            df.shape[0] - numero_amostras_treinamento - numero_amostras_validacao
        )  # % 20%

        df_train = df[0:numero_amostras_treinamento]
        df_validacao = df[
            numero_amostras_treinamento : numero_amostras_treinamento
            + numero_amostras_validacao
        ]
        df_teste = df[
            numero_amostras_treinamento + numero_amostras_validacao :
        ]

        # Features Saida
        output_features = [
            "IRRADIÂNCIA"
        ]
        

        # Persistência
        mae_validacao, series_persistencia_validacao = PersistenciaModel.verificacao_persistencia_mae(
            df_validacao, output_label=output_features[0]
        )
        mae_teste, series_persistencia_teste = PersistenciaModel.verificacao_persistencia_mae(
            df_teste, output_label=output_features[0]
        )

        return mae_validacao, mae_teste, series_persistencia_validacao, series_persistencia_teste

    @staticmethod
    def verificacao_persistencia_mae(
        df: pd.DataFrame, output_label: str
    ):
        df_copy = df.copy()
        df_copy["previsao_persistencia"] = df_copy[output_label].shift(24)
        df_copy["erro_persistencia_mod"] = abs(
            df_copy["previsao_persistencia"] - df_copy[output_label]
        )
        df_copy_without_nan = df_copy.dropna()
        mae = df_copy["erro_persistencia_mod"].mean(axis=0, skipna=True)

        df_series_targets = df_copy_without_nan[["previsao_persistencia", output_label, "Data_Horario"]]
        return mae, df_series_targets



persistencia_model_object = PersistenciaModel()
persistencia_model_object.plot_scatter_previsoes_targets_teste()
persistencia_model_object.plot_scatter_previsoes_targets_validacao()
persistencia_model_object.plot_line_previsoes_targets(series=persistencia_model_object.series_teste, numero_amostras=120)

r_validacao, p_validacao = sp.stats.pearsonr(persistencia_model_object.series_validacao["previsao_persistencia"], persistencia_model_object.series_validacao["IRRADIÂNCIA"])
r_teste, p_teste = sp.stats.pearsonr(persistencia_model_object.series_teste["previsao_persistencia"], persistencia_model_object.series_teste["IRRADIÂNCIA"])

LOGGER = create_logger(debug_mode=False)
LOGGER.info(f"Iniciando Persistencia")
LOGGER.info(f'MAE Validação: {persistencia_model_object.mae_validacao} | MAE Teste: {persistencia_model_object.mae_teste}')
LOGGER.info(f'r_pearson Validação: {r_validacao} | r_pearson Teste: {r_teste}')

print("----")

print("FIM")


