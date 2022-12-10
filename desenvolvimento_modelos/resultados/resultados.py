
import datetime
import gc
from hashlib import sha256
import logging
import os
import pickle
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib
import matplotlib.dates as mdates
from tensorflow import keras
from keras import Model
from keras import backend as K
from dataclasses import dataclass, field
from pandas.util import hash_pandas_object
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE_DIR = "/home/b-rbmp-ideapad/Documents/GitHub/ml-solar-forecaster/"
RF_MODELS_DIR = BASE_DIR + "desenvolvimento_modelos/sklearn_models/models/randomforest/"
GTB_MODELS_DIR = BASE_DIR + "desenvolvimento_modelos/sklearn_models/models/gradienttreeboosting/"
SVR_MODELS_DIR = BASE_DIR + "desenvolvimento_modelos/sklearn_models/models/svr/"
LSTM_MODELS_DIR = BASE_DIR + "desenvolvimento_modelos/neural_models/models/"


class HashableDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)

@dataclass(order=True)
class RegressorStepModel:
    """Classe para guardar informações de cada modelo do problema multioutput"""
    sort_index: int = field(init=False, repr=False)

    step: int
    regressor: RandomForestRegressor | GradientBoostingRegressor
    mae: float
    mse: float
    r2: float
    scaler: Union[MinMaxScaler, StandardScaler]

    def __post_init__(self):
        # sort by step
        self.sort_index = self.step
    
@dataclass(order=True)
class RFRegressorStepModel:
    """Classe para guardar informações de cada modelo do problema multioutput"""
    sort_index: int = field(init=False, repr=False)

    step: int
    regressor: RandomForestRegressor
    mae: float
    mse: float
    r2: float
    scaler: Union[MinMaxScaler, StandardScaler]

    def __post_init__(self):
        # sort by step
        self.sort_index = self.step

@dataclass(order=True)
class GTBRegressorStepModel:
    """Classe para guardar informações de cada modelo do problema multioutput"""
    sort_index: int = field(init=False, repr=False)

    step: int
    regressor: GradientBoostingRegressor
    mae: float
    mse: float
    r2: float
    scaler: Union[MinMaxScaler, StandardScaler]

    def __post_init__(self):
        # sort by step
        self.sort_index = self.step

@dataclass(order=True)
class SVRStepModel:
    """Classe para guardar informações de cada modelo do problema multioutput"""
    sort_index: int = field(init=False, repr=False)

    step: int
    regressor: SVR
    mae: float
    mse: float
    r2: float
    scaler: Union[MinMaxScaler, StandardScaler]

    def __post_init__(self):
        # sort by step
        self.sort_index = self.step

@dataclass(order=True)
class ModelResults:
    """Classe para guardar resultados de cada modelo do problema multioutput"""
    identificador: str
    mae: List[float]
    mse: List[float]
    r2: float
    resultados_finais: List[pd.DataFrame]
    resultados_horarios: List[pd.DataFrame]

    agregado_resultados: pd.DataFrame

    
# Configuracao Logging
def create_logger(debug_mode: bool):
    logging_level: int
    handler: logging.FileHandler
    if not debug_mode:
        handler = logging.FileHandler('resultados_finais.log')
        logging_level = logging.INFO
    else:
        handler = logging.FileHandler('resultados_finais.debug')
        logging_level = logging.DEBUG

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger('ResultadosLogger')
    logger.setLevel(logging_level)
    logger.addHandler(handler)
    return logger

def load_all_rf_models() -> List[RegressorStepModel]:
    models: List[RegressorStepModel] = []
    filenames = [name for name in os.listdir(RF_MODELS_DIR) if os.path.isfile(os.path.join(RF_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(RF_MODELS_DIR, f_name)  # full path
        model: RegressorStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    models.sort()
    return models

def load_all_gtb_models() -> List[RegressorStepModel]:
    models: List[RegressorStepModel] = []
    filenames = [name for name in os.listdir(GTB_MODELS_DIR) if os.path.isfile(os.path.join(GTB_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(GTB_MODELS_DIR, f_name)  # full path
        model: RegressorStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    models.sort()
    return models


def load_all_svr_models() -> List[SVRStepModel]:
    models: List[SVRStepModel] = []
    filenames = [name for name in os.listdir(SVR_MODELS_DIR) if os.path.isfile(os.path.join(SVR_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(SVR_MODELS_DIR, f_name)  # full path
        model: SVRStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    models.sort()
    return models


# Classe para a previsão de persistencia
class PersistenciaForecastObject:

    def __init__(self, hora_problema: int):
        self.hora_problema = hora_problema
        self.identificador = "Persistência"
        self.resultados = self.gerar_forecasts_no_problema_escolhido(hora_problema=hora_problema)
    
    def gerar_forecasts_no_problema_escolhido(self, hora_problema: int):
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
        df_teste = df[
            numero_amostras_treinamento + numero_amostras_validacao :
        ]
        # Features Saida
        output_features = [
            "IRRADIÂNCIA"
        ]

        df_copy_deslocado = df_teste.copy().shift(24)
        df_copy_deslocado["previsao_persistencia"] = df_copy_deslocado[output_features[0]]

        agregado_resultados: pd.DataFrame = pd.DataFrame(data=[], columns=["data", "previsao", "target"])


        agregado_resultados["previsao"] = df_copy_deslocado[output_features[0]]
        agregado_resultados["target"] = df_teste[output_features[0]]
        agregado_resultados["data"] = pd.to_datetime(df_teste["Data_Horario"])
        agregado_resultados.dropna(axis=0, inplace=True)

        indices_hora_problema = agregado_resultados[agregado_resultados["data"].dt.hour == hora_problema]
        indices_hora_problema = indices_hora_problema.iloc[:-1]
        indices_hora_problema = np.array(indices_hora_problema.index)
        agregado_resultados = agregado_resultados.iloc[indices_hora_problema[0]-agregado_resultados.index[0]+1:indices_hora_problema[-1]+1-agregado_resultados.index[0]]
        mae_horario = [0.00 for i in range(0, 24)]
        mse_horario = [0.00 for i in range(0, 24)]
        resultados_finais: List[pd.DataFrame] = [pd.DataFrame(data=[], columns=["data", "previsao", "target"]) for i in range(0, len(indices_hora_problema))] # Array de resultados finais para cada instancia de previsão. Primeira coluna a data das 24 gerações, segunda a previsão e a terceira o resultado esperado
        resultados_horarios: List[pd.DataFrame] = [pd.DataFrame(data=[], columns=["data", "previsao", "target"]) for i in range(0, 24)]

        # 0 a 728, para cada previsão de 24 horas
        for index, value in enumerate(indices_hora_problema):
            if index < len(indices_hora_problema) - 1:
                dataframe_resultado = pd.DataFrame()
                subcorte_dados_previsao = agregado_resultados.loc[np.subtract(indices_hora_problema[index]+1,df.index[0]):np.subtract(indices_hora_problema[index+1]+1,df.index[0])]

                dataframe_resultado["data"] = subcorte_dados_previsao["data"]
                dataframe_resultado["previsao"] = subcorte_dados_previsao["previsao"]
                dataframe_resultado["target"] = subcorte_dados_previsao["target"]
                # Filtra valores negativos
                dataframe_resultado.loc[dataframe_resultado['previsao'] < 0, 'previsao'] = 0
                for indice_horario, prev_horaria in enumerate(subcorte_dados_previsao.values):
                    dados = {}
                    subcorte_local = subcorte_dados_previsao.iloc[indice_horario]
                    dados["data"] = np.array(subcorte_local["data"])
                    dados["previsao"] = np.array(subcorte_local["previsao"])
                    dados["target"] = np.array(subcorte_local["target"])
                    dataframe_horario = pd.DataFrame(data=dados, index=[0])
                    dataframe_horario.loc[dataframe_horario['previsao'] < 0, 'previsao'] = 0
                    dataframe_indexado = resultados_horarios[subcorte_local["data"].hour]
                    resultados_horarios[subcorte_local["data"].hour] = pd.concat(objs=[dataframe_indexado, dataframe_horario], ignore_index=True)
                    
                resultados_finais[index] = dataframe_resultado

        mae_horario = []
        mse_horario = []
        for hora_utc_0, df_hora in enumerate(resultados_horarios):
            mae_horario.append(mean_absolute_error(y_true=df_hora["target"], y_pred=df_hora["previsao"]))
            mse_horario.append(mean_squared_error(y_true=df_hora["target"], y_pred=df_hora["previsao"]))
        
        r2 = r2_score(y_true=agregado_resultados["target"], y_pred=agregado_resultados["previsao"])

        return ModelResults(mae=mae_horario, mse=mse_horario, r2=r2, resultados_finais=resultados_finais, resultados_horarios=resultados_horarios, agregado_resultados=agregado_resultados, identificador=self.identificador)
    

# Classe para os Forecasts Simples LSTM (com apenas 1 ticker)
class LSTMForecastObject:

    def __init__(self, arquitetura: str, hora_problema: int, hora_inicio_periodo_solar: int, hora_fim_periodo_solar: int):
        self.hora_problema = hora_problema
        self.hora_inicio_periodo_solar = hora_inicio_periodo_solar
        self.hora_fim_periodo_solar = hora_fim_periodo_solar
        self.arquitetura = arquitetura
        self.resultados = self.gerar_forecasts_no_problema_escolhido(arquitetura, hora_inicio_periodo_solar=hora_inicio_periodo_solar, hora_fim_periodo_solar=hora_fim_periodo_solar)


    @staticmethod
    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
        
    @staticmethod
    def prod_normalizacao_stock_dfs(df_target: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
        df_copy = df_target.copy()


        df_index = df_copy.index
        df_copy.replace(-np.inf, 0, inplace=True)
        df_copy.replace(np.inf, 0, inplace=True)

        
        df_horas = df_copy["hora_num"]
        df_copy = df_copy.drop(labels=["hora_num"], axis=1)
        df_labels = df_copy.columns
        df_copy = scaler.transform(df_copy)
        df_copy = pd.DataFrame(data=df_copy, columns=df_labels, index=df_index)
        df_copy["hora_num"] = df_horas
        df_copy.dropna(inplace=True)

        df_copy[df_copy.select_dtypes(np.float64).columns] = df_copy.select_dtypes(np.float64).astype(np.float32)

        return df_copy

    def split_sequence_problema_escolhido(self,
        sequence,
        look_back,
        padding_between_lookback_forecast,
        forecast_horizon,
        labels_input_measurements,
        labels_input_forecasts,
        labels_output,
        hora_geracao_previsoes,
    ):
        # horarios 7am UTC-0
        X_measurements, X_forecasts, y, indices = list(), list(), list(), list()
        for i in range(len(sequence)):
            lag_end = i + look_back
            forecast_end = (
                lag_end + forecast_horizon
            )
            if forecast_end > len(sequence):
                break
            sequence_last_hour_measurement_item = sequence.iloc[lag_end-1]
            sequence_last_hour_measurement_num = sequence_last_hour_measurement_item["hora_num"]
            if (sequence_last_hour_measurement_num == hora_geracao_previsoes):
                seq_x_measurements, seq_x_forecasts, seq_y = (
                    sequence[i:lag_end][list(labels_input_measurements)],
                    sequence[i:forecast_end][list(labels_input_forecasts)],
                    sequence[lag_end - 1 + padding_between_lookback_forecast:forecast_end][list(labels_output)],
                )
                X_measurements.append(seq_x_measurements)
                X_forecasts.append(seq_x_forecasts)
                y.append(seq_y)
                indices.append(sequence.index[lag_end-1])
        X_measurements = np.array(X_measurements)
        X_forecasts = np.array(X_forecasts)
        X_array = np.full((X_measurements.shape[0], max(X_measurements.shape[1], X_forecasts.shape[1]), (X_measurements.shape[2]+X_forecasts.shape[2])), -1, dtype=np.float64)
        X_array[:, :, :-1] = X_forecasts
        X_array[:, :X_measurements.shape[1], X_array.shape[2]-X_measurements.shape[2]:] = X_measurements
        X = np.array(X_array)

        
        return X, np.array(y), indices

        
    def gerar_forecasts_no_problema_escolhido(self, arquitetura: str, hora_inicio_periodo_solar: int, hora_fim_periodo_solar: int):
        df = pd.read_csv(f"{BASE_DIR}dados/pre_processado/salvador.csv")
        regressor: Model = keras.models.load_model(LSTM_MODELS_DIR+arquitetura+"/modelo_treinado.keras", custom_objects={"coeff_determination": LSTMForecastObject.coeff_determination})
        scaler: MinMaxScaler = joblib.load(LSTM_MODELS_DIR+arquitetura+f"/scaler.gz")

        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

        # Adição de Coluna para Radiação t-1
        # Separacao das Features que vão até t (todas exceto radiação) e features que vão até t-1 para entrada.
        # Para isso será criado uma nova coluna de radiação que é a radiação atrasada de 1h
        # nome_coluna_radiacao_shifted = "RADIACAO t-1"
        # df[nome_coluna_radiacao_shifted] = df["RADIACAO GLOBAL(Kj/m²)"].shift(1)
        # df.fillna(method="bfill", inplace=True)

        # Dropa colunas não utilizadas
        df.drop(["Unnamed: 0"], axis=1, inplace=True)

        # Transformações para adicionar colunas de hora, dia, mês e ano
        series_data = pd.to_datetime(df["Data_Horario"], infer_datetime_format=True)

        # Criando uma Feature Cíclica representado a posição do Momento no Ano
        s = 24 * 60 * 60 # Segundos no ano 
        year = (365.25) * s
        df["timestamp"] = series_data.values.astype('int64') // 10**9
        df["ano_cos"] = [np.cos((x) * (2 * np.pi / year)) for x in df["timestamp"]]
        df["ano_sin"] = [np.sin((x) * (2 * np.pi / year)) for x in df["timestamp"]]
        df.drop(labels=["timestamp"], axis=1, inplace=True)

        # Criando uma Feature Cíclica representado a posição da Hora no Dia
        df["hora_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in series_data.dt.hour]
        df["hora_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in series_data.dt.hour]
        df["hora_num"] = series_data.dt.hour
        df_sem_data_horario = df.drop(labels=["Data_Horario"], axis=1)

        # Normalização 
        df_normalizado = LSTMForecastObject.prod_normalizacao_stock_dfs(df_target=df_sem_data_horario, scaler=scaler)

        # Preparação dos Dados

        # Train-Validation-Test split
        numero_amostras_treinamento = int(0.6 * df_sem_data_horario.shape[0])  # 60%
        numero_amostras_validacao = int(0.2 * df_sem_data_horario.shape[0])  # 20%
        numero_amostras_teste = (
            df_sem_data_horario.shape[0] - numero_amostras_treinamento - numero_amostras_validacao
        )  # % 20%

        df_teste = df_normalizado[
            numero_amostras_treinamento + numero_amostras_validacao :
        ]

        input_features_forecast = [
            "ano_cos",
            "ano_sin",
            "hora_cos",
            "hora_sin",
            "RH2M",
            "WD50M",
            "PSC",
        ]

        input_measurements = [
            "IRRADIÂNCIA"
        ]

        # Features Saida
        output_features = [
            "IRRADIÂNCIA"
        ]

        LOOK_BACK = 24
        forecast_horizon = 24
        PADDING_BETWEEN_LOOK_BACK_FORECAST = 1

    
        input_test, output_test, indices = self.split_sequence_problema_escolhido(
            HashableDataFrame(df_teste),
            look_back=LOOK_BACK,
            forecast_horizon=forecast_horizon,
            padding_between_lookback_forecast=PADDING_BETWEEN_LOOK_BACK_FORECAST,
            labels_input_measurements=tuple(input_measurements),
            labels_input_forecasts=tuple(input_features_forecast),
            labels_output=tuple(output_features),
            hora_geracao_previsoes=self.hora_problema,
        )

        # Scaler apenas para irradiancia
        scale = MinMaxScaler()
        scale.min_, scale.scale_ = scaler.min_[10], scaler.scale_[10]
        # Gera previsões para cada step horário definido no problema
        predicao = regressor.predict(input_test)
        predicao_scaled = scale.inverse_transform(np.array(predicao).reshape(-1, 1)).flatten().reshape(-1, 24)
        resultados_finais: List[pd.DataFrame] = [pd.DataFrame(data=[], columns=["data", "previsao", "target"]) for i in range(0, len(predicao))] # Array de resultados finais para cada instancia de previsão. Primeira coluna a data das 24 gerações, segunda a previsão e a terceira o resultado esperado
        resultados_horarios: List[pd.DataFrame] = [pd.DataFrame(data=[], columns=["data", "previsao", "target"]) for i in range(0, 24)]
        agregado_resultados: pd.DataFrame = pd.DataFrame(data=[], columns=["data", "previsao", "target"])
        for index, pred in enumerate(predicao_scaled):
            if (index < len(predicao_scaled) - 1):
                dataframe_resultado = pd.DataFrame()
                subcorte_dados_previsao = df.iloc[np.subtract(indices[index]+1,df.index[0]):np.subtract(indices[index+1]+1,df.index[0])]

                dataframe_resultado["data"] = np.array(subcorte_dados_previsao["Data_Horario"])
                dataframe_resultado["previsao"] = pred
                dataframe_resultado["target"] = np.array(subcorte_dados_previsao["IRRADIÂNCIA"])
                # Filtra valores negativos
                dataframe_resultado.loc[dataframe_resultado['previsao'] < 0, 'previsao'] = 0
                # Filtra valores fora do periodo solar
                datetime_index = pd.to_datetime(dataframe_resultado['data']).dt.hour
                dataframe_resultado.loc[datetime_index <= hora_inicio_periodo_solar, 'previsao'] = 0
                dataframe_resultado.loc[datetime_index >= hora_fim_periodo_solar, 'previsao'] = 0

                for indice_horario, prev_horaria in enumerate(pred):
                    dados = {}
                    dados["data"] = np.array(subcorte_dados_previsao["Data_Horario"].iloc[indice_horario])
                    dados["previsao"] = np.array(prev_horaria)
                    dados["target"] = np.array(subcorte_dados_previsao["IRRADIÂNCIA"].iloc[indice_horario])
                    dataframe_horario = pd.DataFrame(data=dados, index=[0])
                    # Filtra valores negativos
                    dataframe_horario.loc[dataframe_horario['previsao'] < 0, 'previsao'] = 0
                    # Filtra valores fora do periodo solar
                    datetime_index = datetime.datetime.strptime(subcorte_dados_previsao["Data_Horario"].iloc[indice_horario], "%Y-%m-%d %H:%M:%S").hour
                    if datetime_index <= hora_inicio_periodo_solar or datetime_index >= hora_fim_periodo_solar:
                        dataframe_horario["previsao"] = pd.Series([0])
                    dataframe_indexado = resultados_horarios[datetime.datetime.strptime(subcorte_dados_previsao["Data_Horario"].iloc[indice_horario], "%Y-%m-%d %H:%M:%S").hour]
                    resultados_horarios[datetime.datetime.strptime(subcorte_dados_previsao["Data_Horario"].iloc[indice_horario], "%Y-%m-%d %H:%M:%S").hour] = pd.concat(objs=[dataframe_indexado, dataframe_horario], ignore_index=True)
                    
                agregado_resultados = pd.concat(objs=[agregado_resultados, dataframe_resultado], ignore_index=True)
                resultados_finais[index] = dataframe_resultado

        mae_horario = []
        mse_horario = []
        for hora_utc_0, df_hora in enumerate(resultados_horarios):
            mae_horario.append(mean_absolute_error(y_true=df_hora["target"], y_pred=df_hora["previsao"]))
            mse_horario.append(mean_squared_error(y_true=df_hora["target"], y_pred=df_hora["previsao"]))
        
        r2 = r2_score(y_true=agregado_resultados["target"], y_pred=agregado_resultados["previsao"])

        return ModelResults(mae=mae_horario, mse=mse_horario, r2=r2, resultados_finais=resultados_finais, resultados_horarios=resultados_horarios, agregado_resultados=agregado_resultados, identificador=self.arquitetura)

class SKLearnForecastObject:

    def __init__(self, tipo: str, hora_problema: int, hora_inicio_periodo_solar: int, hora_fim_periodo_solar: int):
        self.hora_problema = hora_problema
        self.hora_inicio_periodo_solar = hora_inicio_periodo_solar
        self.hora_fim_periodo_solar = hora_fim_periodo_solar
        self.tipo = tipo
        self.resultados = self.gerar_forecasts_no_problema_escolhido(tipo=tipo, hora_inicio_periodo_solar=hora_inicio_periodo_solar, hora_fim_periodo_solar=hora_fim_periodo_solar)
    
    # Standardization / Normalization
    @staticmethod
    def prod_normalizacao_stock_dfs(df_target: pd.DataFrame, scaler: MinMaxScaler, label_target: str | None = "IRRADIÂNCIA") -> pd.DataFrame:
        
        df_copy = df_target.copy()

        # Não normaliza o target
        if label_target is not None:
            df_target = df_copy[label_target]

        df_data_horario = df_copy["Data_Horario"]
        df_index = df_copy.index
        df_copy.replace(-np.inf, 0, inplace=True)
        df_copy.replace(np.inf, 0, inplace=True)
        df_copy.drop(labels=[label_target, "Data_Horario"], axis=1, inplace=True)
        df_horas = df_copy["hora_num"]
        df_copy = df_copy.drop(labels=["hora_num"], axis=1)
        df_labels = df_copy.columns

        df_copy = scaler.transform(df_copy)
        df_copy = pd.DataFrame(data=df_copy, columns=df_labels, index=df_index)
        df_copy["hora_num"] = df_horas
        if label_target is not None:
            df_copy[label_target] = df_target
        df_copy["Data_Horario"] = df_data_horario
        df_copy.dropna(inplace=True)

        df_copy[df_copy.select_dtypes(np.float64).columns] = df_copy.select_dtypes(np.float64).astype(np.float32)

        return df_copy

    def gerar_dados_problema_supervisionado(forecast_data: HashableDataFrame, measurements_data: HashableDataFrame, output_data: HashableDataFrame, n_in: int, n_out: int, data_horario_df: HashableDataFrame, hora_problema: int) -> List[HashableDataFrame]:
        lista_dfs_outputs: List[HashableDataFrame] = []
        df_target = output_data.copy()
        df_a_deslocar_futuro = pd.concat([forecast_data], axis=1)
        df_a_deslocar_passado = pd.concat([forecast_data, measurements_data], axis=1)
        df_deslocado_passado: pd.DataFrame = df_a_deslocar_passado.copy()
        colunas_temporais = ["hora_sin", "hora_cos", "ano_cos", "ano_sin"]
        df_info_temporal = df_a_deslocar_futuro[colunas_temporais]
        df_a_deslocar_passado.drop(labels=colunas_temporais, axis=1, inplace=True)
        df_deslocado_passado.drop(labels=colunas_temporais, axis=1, inplace=True)
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            df_a_deslocar = df_a_deslocar_passado.copy()
            df_deslocado = df_a_deslocar.shift(i)
            df_deslocado = df_deslocado.add_suffix(suffix=f"(t-{i})")
            df_deslocado_passado = pd.concat([df_deslocado_passado, df_deslocado], axis=1)


        # forecast sequence (t, t+1, ... t+n)
        df_a_deslocar_futuro.drop(labels=colunas_temporais, axis=1, inplace=True)
        df_deslocado_futuro = pd.DataFrame(dtype=np.float32)
        data_horario_df_datelike = pd.DataFrame({"Data_Horario": pd.to_datetime(data_horario_df["Data_Horario"])})
        for i in range(1, n_out+1):
            df_a_deslocar = df_a_deslocar_futuro.copy()
            df_deslocado = df_a_deslocar.shift(-i)
            df_deslocado = df_deslocado.add_suffix(suffix=f"(t+{i})")
            df_deslocado_futuro = pd.concat([df_deslocado_futuro, df_deslocado], axis=1)
            df_inputs = pd.concat([df_deslocado_passado, df_deslocado_futuro], axis=1)
            df_output = df_target.shift(-i)
            df_output.rename(columns={"IRRADIÂNCIA": "IRRADIÂNCIA TARGET"}, inplace=True)
            df_temporal_target = df_info_temporal.shift(-i)
            df_future_model = pd.concat([df_inputs, df_output, df_temporal_target, data_horario_df_datelike], axis=1)
            df_future_model.dropna(inplace=True, axis=0)
            df_future_model = df_future_model.sort_index(axis=1)
            df_future_model = df_future_model[df_future_model["Data_Horario"].dt.hour == hora_problema]
            lista_dfs_outputs.append(df_future_model)



        return lista_dfs_outputs

    def gerar_forecasts_no_problema_escolhido(self, tipo: str, hora_inicio_periodo_solar: int, hora_fim_periodo_solar: int):
        df = pd.read_csv(f"{BASE_DIR}dados/pre_processado/salvador.csv")
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

        # Adição de Coluna para Radiação t-1
        # Separacao das Features que vão até t (todas exceto radiação) e features que vão até t-1 para entrada.
        # Para isso será criado uma nova coluna de radiação que é a radiação atrasada de 1h
        # nome_coluna_radiacao_shifted = "RADIACAO t-1"
        # df[nome_coluna_radiacao_shifted] = df["RADIACAO GLOBAL(Kj/m²)"].shift(1)
        # df.fillna(method="bfill", inplace=True)

        # Dropa colunas não utilizadas
        df.drop(["Unnamed: 0"], axis=1, inplace=True)

        # Transformações para adicionar colunas de hora, dia, mês e ano
        series_data = pd.to_datetime(df["Data_Horario"], infer_datetime_format=True)

        # Criando uma Feature Cíclica representado a posição do Momento no Ano
        s = 24 * 60 * 60 # Segundos no ano 
        year = (365.25) * s
        df["timestamp"] = series_data.values.astype('int64') // 10**9
        df["ano_cos"] = [np.cos((x) * (2 * np.pi / year)) for x in df["timestamp"]]
        df["ano_sin"] = [np.sin((x) * (2 * np.pi / year)) for x in df["timestamp"]]
        df.drop(labels=["timestamp"], axis=1, inplace=True)

        # Criando uma Feature Cíclica representado a posição da Hora no Dia
        df["hora_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in series_data.dt.hour]
        df["hora_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in series_data.dt.hour]
        df["hora_num"] = series_data.dt.hour


        step_models: List[SVRStepModel | RFRegressorStepModel | GTBRegressorStepModel] = []
        if tipo == "randomforest":
            step_models = load_all_rf_models()
        elif tipo == "gtb":
            step_models = load_all_gtb_models()
        elif tipo =="svr":
            step_models = load_all_svr_models()

        scaler = step_models[0].scaler


        # Normalização
        df_target_normalized = SKLearnForecastObject.prod_normalizacao_stock_dfs(df_target=df, scaler=scaler)

        # Preparação dos Dados

        # Train-Validation-Test split
        numero_amostras_treinamento = int(0.6 * df_target_normalized.shape[0])  # 60%
        numero_amostras_validacao = int(0.2 * df_target_normalized.shape[0])  # 20%
        numero_amostras_teste = (
            df_target_normalized.shape[0] - numero_amostras_treinamento - numero_amostras_validacao
        )  # % 20%

        df_teste = df_target_normalized[
            numero_amostras_treinamento + numero_amostras_validacao :
        ]

        input_features_forecast = [
            "ano_cos",
            "ano_sin",
            "hora_cos",
            "hora_sin",
            "RH2M",
            "WD50M",
            "PRECTOTCORR",
            "T2M",
            "WD10M",
            "WS50M",
            "PSC",
        ]

        input_measurements = [
            "IRRADIÂNCIA"
        ]

        # Features Saida
        output_features = [
            "IRRADIÂNCIA"
        ]


        in_n_measures=24
        out_n_measures=24

        data_multioutput_supervised = SKLearnForecastObject.gerar_dados_problema_supervisionado(forecast_data=HashableDataFrame(df_teste[input_features_forecast]), measurements_data=HashableDataFrame(df_teste[input_measurements]), output_data=HashableDataFrame(df_teste.iloc[in_n_measures:][output_features]), n_in=in_n_measures, n_out=out_n_measures, data_horario_df=HashableDataFrame(df_teste["Data_Horario"]), hora_problema=self.hora_problema)

        target_output_after_treatment = "IRRADIÂNCIA TARGET"
        range_models = range(0, len(data_multioutput_supervised))
        resultados_finais: List[pd.DataFrame] = [pd.DataFrame(data=[], columns=["data", "previsao", "target"]) for i in range(0, len(data_multioutput_supervised[-1]))] # Array de resultados finais para cada instancia de previsão. Primeira coluna a data das 24 gerações, segunda a previsão e a terceira o resultado esperado
        resultados_horarios: List[pd.DataFrame] = [pd.DataFrame(data=[], columns=["data", "previsao", "target"]) for i in range(0, len(data_multioutput_supervised))]
        agregado_resultados: pd.DataFrame = pd.DataFrame(data=[], columns=["data", "previsao", "target"])
        mae_horario = [0.00 for i in range(0, len(data_multioutput_supervised))]
        mse_horario = [0.00 for i in range(0, len(data_multioutput_supervised))]

        for i in range_models:
            df_data = data_multioutput_supervised[i]

            df_data_limitado = df_data.iloc[:-1]
            alvos = df_data_limitado["IRRADIÂNCIA TARGET"]
            df_inputs = df_data.drop(labels=[target_output_after_treatment, "Data_Horario"], axis=1)
            modelo = step_models[i]
            

            Y_pred = modelo.regressor.predict(np.array(df_inputs))
            Y_pred = Y_pred[:-1] # Descarta o ultimo por não possuir target (futuro)
            
            step = i + 1

            datetime_data_horario = pd.to_datetime(df_data_limitado["Data_Horario"])
            data_horario_corrigido = datetime_data_horario + datetime.timedelta(hours=step)


            dados = {}
            dados["data"] = np.array(data_horario_corrigido)
            dados["previsao"] = np.array(Y_pred)
            dados["target"] = np.array(df_data_limitado["IRRADIÂNCIA TARGET"])
            dataframe_horario = pd.DataFrame(data=dados)
            
            # Filtra valores negativos
            dataframe_horario.loc[dataframe_horario['previsao'] < 0, 'previsao'] = 0
            
            # Filtra valores fora do horário solar   
            datetime_index = pd.to_datetime(dataframe_horario['data']).dt.hour
            dataframe_horario.loc[datetime_index <= hora_inicio_periodo_solar, 'previsao'] = 0
            dataframe_horario.loc[datetime_index >= hora_fim_periodo_solar, 'previsao'] = 0

            agregado_resultados = pd.concat(objs=[agregado_resultados, dataframe_horario], axis=0)

            hora_step = (datetime_data_horario.iloc[0] + datetime.timedelta(hours=step)).hour
            resultados_horarios[hora_step] = dataframe_horario
            mae_horario[hora_step] = mean_absolute_error(y_true=np.array(alvos), y_pred=np.array(dataframe_horario["previsao"]))
            mse_horario[hora_step] = mean_squared_error(y_true=np.array(alvos), y_pred=np.array(dataframe_horario["previsao"]))

            for num_previsao, previsao in enumerate(dataframe_horario.values):
                resultados_finais[num_previsao].loc[len(resultados_finais[num_previsao])] = previsao

        # Indice temporario para preenchimento
        for i_resultado_final in range(0, len(resultados_finais)):
            resultados_finais[i_resultado_final] = resultados_finais[i_resultado_final].set_index("data", drop=True).sort_index()
        
        agregado_resultados = agregado_resultados.set_index("data", drop=True).sort_index()

        # Reseta indices
        for i_resultado_final in range(0, len(resultados_finais)):
            resultados_finais[i_resultado_final] = resultados_finais[i_resultado_final].reset_index()

        agregado_resultados = agregado_resultados.reset_index()

        r2 = r2_score(y_true=agregado_resultados["target"], y_pred=agregado_resultados["previsao"])

        ############################ FILTRA ALI EM CIMA COM BASE NO INDEX PRA SER UTC-0 = 7, PRECISA RETORNAR EM GERAR_DADOS O HORARIO Q NEM NO LSTM
        return ModelResults(mae=mae_horario, mse=mse_horario, r2=r2, resultados_finais=resultados_finais, resultados_horarios=resultados_horarios, agregado_resultados=agregado_resultados, identificador=self.tipo)


def histograma_erros_hora_multiplot(modelos_results: List[ModelResults], hora_utc_0 = 15):
    # Gera um multiplot com 2 colunas
    n_cols = 2
    n_rows = math.ceil(len(modelos_results)/n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 50), sharex=True, sharey=True)

    resultados = pd.DataFrame()
    
    for modelo_result in modelos_results:
        resultados_agregados = modelo_result.agregado_resultados
        resultados_agregados_filtrado = resultados_agregados[pd.to_datetime(resultados_agregados["data"]).dt.hour == hora_utc_0]
        resultados_agregados_filtrado.set_index("data", inplace=True, drop=True)
        resultados[modelo_result.identificador] = np.abs(np.subtract(resultados_agregados_filtrado["previsao"], resultados_agregados_filtrado["target"]))
    
    i = 0
    for doubleaxis in axes:
        for axis in doubleaxis:
            if i >= len(modelos_results):
                break
            sns.histplot(data=resultados, x=resultados.columns[i], kde=True, ax=axis, bins=20)
            axis.set_ylabel("")
            axis.set_xlabel("")
            axis.set_title(label=resultados.columns[i], y=0.78, fontsize=22)
            axis.tick_params(axis='both', which='both', labelsize=18)
            i = i+1
    fig.supxlabel('Erro Absoluto (W/m²)', y=0.02, fontsize=22)
    fig.supylabel('N° Ocorrências', x=0.06, fontsize=22)
    plt.show()

def kde_erros_hora(modelos_results: List[ModelResults], hora_utc_0 = 15):
    resultados = pd.DataFrame()
    
    colunas = []
    for modelo_result in modelos_results:
        resultados_agregados = modelo_result.agregado_resultados
        resultados_agregados_filtrado = resultados_agregados[pd.to_datetime(resultados_agregados["data"]).dt.hour == hora_utc_0]
        resultados_agregados_filtrado.set_index("data", inplace=True, drop=True)
        resultados[modelo_result.identificador] = np.subtract(resultados_agregados_filtrado["previsao"], resultados_agregados_filtrado["target"])
        colunas.append(modelo_result.identificador)

    plt.figure()
    sns.set(font_scale=1.5)
    sns.kdeplot(data=resultados, legend=True)


    plt.xlabel("Erro (W/m²)", fontsize=22)
    plt.ylabel("Densidade de Probabilidade", fontsize=22)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.grid()
    plt.show()


def barplot_horizontal_r2(modelos_results: List[ModelResults]):
    resultados = pd.DataFrame()
    
    for modelo_result in modelos_results:
        resultados[modelo_result.identificador] = [modelo_result.r2]

    plt.figure(figsize=(14, 5))
    ax = sns.barplot(data=resultados, orient="h")
    for i in ax.containers:
        ax.bar_label(i, fontsize=17)

    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    plt.xlabel("Coeficiente de Determinação (R²)", fontsize=20)
    plt.tight_layout()

    plt.show()

def barplot_horizontal_mae_medio(modelos_results: List[ModelResults]):
    resultados = pd.DataFrame()
    
    for modelo_result in modelos_results:
        resultados[modelo_result.identificador] = [round(np.mean(modelo_result.mae), 2)]

    plt.figure(figsize=(14, 5))
    ax = sns.barplot(data=resultados, orient="h")
    for i in ax.containers:
        ax.bar_label(i, fontsize=17)

    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    plt.xlabel("Erro Absoluto Médio - MAE (W/m²)", fontsize=20)
    plt.tight_layout()
    plt.show()

def lineplot_mae_horario(modelos_results: List[ModelResults]):
    resultados = pd.DataFrame()
    
    colunas = []
    for modelo_result in modelos_results:
        resultados[modelo_result.identificador] = modelo_result.mae
        colunas.append(modelo_result.identificador)
    
    plt.figure()
    sns.lineplot(data=resultados, markers=True, dashes=False, legend=True)

    plt.xlabel("Hora UTC-0", fontsize=22)
    plt.ylabel("Erro Absoluto Médio - MAE (W/m²)", fontsize=22)
    indices = [i for i in range(0, len(modelos_results[0].mae))]
    plt.xticks(indices)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.grid()
    plt.legend(fontsize=18)
    plt.show()

def previsao_plot(modelos_results: List[ModelResults], data_inicio: str, data_fim: str):
    resultados = pd.DataFrame()
    

    for modelo_result in modelos_results:
        resultados_agregados = modelo_result.agregado_resultados
        resultados_agregados_filtrado = resultados_agregados[(pd.to_datetime(resultados_agregados['data']) >= (datetime.datetime.strptime(data_inicio, "%Y-%m-%d") + datetime.timedelta(hours=8))) & (pd.to_datetime(resultados_agregados['data']) <= (datetime.datetime.strptime(data_fim, "%Y-%m-%d") + datetime.timedelta(hours=7)))]
        resultados_agregados_filtrado.set_index("data", inplace=True, drop=True)
        resultados[modelo_result.identificador] = resultados_agregados_filtrado["previsao"]
    
    resultados["target"] = resultados_agregados_filtrado["target"]

    fig, ax = plt.subplots(figsize=(18, 12))
    #indices = [x for x in np.array(pd.to_datetime(resultados.index).strftime("%d/%m/%Y %H:%M"))]
    ax.plot(pd.to_datetime(resultados.index).values, resultados["target"], linestyle='dashed', marker=".")
    ax.plot(pd.to_datetime(resultados.index).values, resultados.drop("target", axis=1), marker=".")


    plt.xlabel("Data e Hora UTC-0", fontsize=22)
    plt.ylabel("Irradiância Solar Horária (W/m²)", fontsize=22)
    
    # Format the x axis
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    date_form = mdates.DateFormatter('%d/%m/%Y %H:%M')
    ax.xaxis.set_major_formatter(date_form)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    plt.grid()
    plt.legend(["Alvo"] + list(resultados.columns), fontsize=18)
    plt.show()
    
def moving_average_irradiancia():
    sns.set_context("talk")
    df = pd.read_csv(f"{BASE_DIR}dados/pre_processado/salvador.csv")
    n_dias = 100
    df["Data_Horario"] = pd.to_datetime(df["Data_Horario"])
    df = df[df["Data_Horario"].dt.hour == 15]
    df['irradiância_rolling'] = df["IRRADIÂNCIA"].rolling(math.ceil(100)).mean()
    df.dropna(axis=0, inplace=True)
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.lineplot(x="Data_Horario",y="IRRADIÂNCIA",label="Irradiância Horária", data=df, ax=ax)
    sns.lineplot(x="Data_Horario",y="irradiância_rolling", label=f"Irradiância - Média Móvel de {n_dias} dias", data=df, ax=ax)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    date_form = mdates.DateFormatter('%m/%Y')
    ax.xaxis.set_major_formatter(date_form)
    plt.xlabel("Data", fontsize=22)
    plt.ylabel("Irradiãncia Horária (W/m²)", fontsize=22)
    plt.legend(fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.show()

persistencia_model = PersistenciaForecastObject(hora_problema=7)
vanilla_lstm_model = LSTMForecastObject("vanilla_lstm", hora_problema=7, hora_inicio_periodo_solar=8, hora_fim_periodo_solar=22)
encoder_decoder_lstm_model = LSTMForecastObject("encoder_decoder_lstm", hora_problema=7, hora_inicio_periodo_solar=8, hora_fim_periodo_solar=22)
encoder_decoder_cnn_lstm_model = LSTMForecastObject("encoder_decoder_cnn_lstm", hora_problema=7, hora_inicio_periodo_solar=8, hora_fim_periodo_solar=22)
rf_model = SKLearnForecastObject("randomforest", hora_problema=7, hora_inicio_periodo_solar=8, hora_fim_periodo_solar=22)
svr_model = SKLearnForecastObject("svr", hora_problema=7, hora_inicio_periodo_solar=8, hora_fim_periodo_solar=22)
gtb_model = SKLearnForecastObject("gtb", hora_problema=7, hora_inicio_periodo_solar=8, hora_fim_periodo_solar=22)

modelos_results=[vanilla_lstm_model.resultados, encoder_decoder_lstm_model.resultados, encoder_decoder_cnn_lstm_model.resultados, rf_model.resultados, svr_model.resultados, gtb_model.resultados, persistencia_model.resultados]
#Histograma de erros para cada modelo (multiplot) em hora especifica
histograma_erros_hora_multiplot(modelos_results=modelos_results, hora_utc_0=12)
histograma_erros_hora_multiplot(modelos_results=modelos_results, hora_utc_0=15)
histograma_erros_hora_multiplot(modelos_results=modelos_results, hora_utc_0=20)
# KDE em hora especifica
kde_erros_hora(modelos_results=modelos_results, hora_utc_0=12)
kde_erros_hora(modelos_results=modelos_results, hora_utc_0=15)
kde_erros_hora(modelos_results=modelos_results, hora_utc_0=20)

# Barplot horizontal comparando r2 (de tudo), mae médio para cada modelo, incluindo persistencia
barplot_horizontal_mae_medio(modelos_results=modelos_results)
barplot_horizontal_r2(modelos_results=modelos_results)
# Lineplot comparando mae por hora pra cada modelo, incluindo persistencia
lineplot_mae_horario(modelos_results)
# Previsão de ultimos 4 dias para cada modelo, uns 2-3 exemplos de periodos diferentes (ex: chuvoso etc...)
previsao_plot(modelos_results, data_inicio="2022-08-17", data_fim="2022-08-19")
previsao_plot(modelos_results, data_inicio="2021-12-22", data_fim="2021-12-24")
previsao_plot(modelos_results, data_inicio="2021-03-01", data_fim="2021-03-03")

# Moving average irradiãncia
moving_average_irradiancia()
print("FIM")
# Potencial ensemble somando tudo (talvez um modelo treinado? - talvez como trabalhos futuros)