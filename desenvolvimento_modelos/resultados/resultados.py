
import gc
from hashlib import sha256
import logging
import os
import pickle
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import math
import joblib
from tensorflow import keras
from keras import Model
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
class RFRegressorStepModel:
    """Classe para guardar informações de cada modelo do problema multioutput"""
    sort_index: int = field(init=False, repr=False)

    step: int
    regressor: RandomForestRegressor
    mae: float
    mse: float
    r2: float
    y_pred: List[float]
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
    y_pred: List[float]
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
    y_pred: List[float]
    scaler: Union[MinMaxScaler, StandardScaler]

    def __post_init__(self):
        # sort by step
        self.sort_index = self.step

@dataclass(order=True)
class LSTMModel:
    """Classe para guardar informações de cada modelo do problema multioutput"""
    regressor: Model
    mae: List[float]
    mse: List[float]
    r2: List[float]
    y_pred: List[List[float]]


    
def gerar_dados_problema_supervisionado(forecast_data: HashableDataFrame, measurements_data: HashableDataFrame, output_data: HashableDataFrame, n_in: int, n_out: int) -> List[HashableDataFrame]:
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
    for i in range(1, n_out+1):
        df_a_deslocar = df_a_deslocar_futuro.copy()
        df_deslocado = df_a_deslocar.shift(-i)
        df_deslocado = df_deslocado.add_suffix(suffix=f"(t+{i})")
        df_deslocado_futuro = pd.concat([df_deslocado_futuro, df_deslocado], axis=1)
        df_inputs = pd.concat([df_deslocado_passado, df_deslocado_futuro], axis=1)
        df_output = df_target.shift(-i)
        df_output.rename(columns={"IRRADIÂNCIA": "IRRADIÂNCIA TARGET"}, inplace=True)
        df_temporal_target = df_info_temporal.shift(-i)
        df_future_model = pd.concat([df_inputs, df_output, df_temporal_target], axis=1)
        df_future_model.dropna(inplace=True, axis=0)
        df_future_model = df_future_model.sort_index(axis=1)
        lista_dfs_outputs.append(df_future_model)



    return lista_dfs_outputs

def split_sequence(
    sequence,
    look_back,
    padding_between_lookback_forecast,
    forecast_horizon,
    labels_input_measurements,
    labels_input_forecasts,
    labels_output,
):
    X_measurements, X_forecasts, y = list(), list(), list()
    for i in range(len(sequence)):
        lag_end = i + look_back
        forecast_end = (
            lag_end + forecast_horizon
        )
        if forecast_end > len(sequence):
            break
        seq_x_measurements, seq_x_forecasts, seq_y = (
            sequence[i:lag_end][list(labels_input_measurements)],
            sequence[i:forecast_end][list(labels_input_forecasts)],
            sequence[lag_end - 1 + padding_between_lookback_forecast:forecast_end][list(labels_output)],
        )
        X_measurements.append(seq_x_measurements)
        X_forecasts.append(seq_x_forecasts)
        y.append(seq_y)
    X_measurements = np.array(X_measurements)
    X_forecasts = np.array(X_forecasts)
    X_array = np.full((X_measurements.shape[0], max(X_measurements.shape[1], X_forecasts.shape[1]), (X_measurements.shape[2]+X_forecasts.shape[2])), -1, dtype=np.float64)
    X_array[:, :, :-1] = X_forecasts
    X_array[:, :X_measurements.shape[1], X_array.shape[2]-X_measurements.shape[2]:] = X_measurements
    X = np.array(X_array)

    return X, np.array(y)

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

def load_all_rf_models() -> List[RFRegressorStepModel]:
    models: List[RFRegressorStepModel] = []
    filenames = [name for name in os.listdir(RF_MODELS_DIR) if os.path.isfile(os.path.join(RF_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(RF_MODELS_DIR, f_name)  # full path
        model: RFRegressorStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    return models

def load_all_gtb_models() -> List[GTBRegressorStepModel]:
    models: List[GTBRegressorStepModel] = []
    filenames = [name for name in os.listdir(GTB_MODELS_DIR) if os.path.isfile(os.path.join(GTB_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(GTB_MODELS_DIR, f_name)  # full path
        model: GTBRegressorStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    return models


def load_all_svr_models() -> List[SVRStepModel]:
    models: List[SVRStepModel] = []
    filenames = [name for name in os.listdir(SVR_MODELS_DIR) if os.path.isfile(os.path.join(SVR_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(SVR_MODELS_DIR, f_name)  # full path
        model: SVRStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    return models

# Classe para os Forecasts Simples (com apenas 1 ticker)
class LSTMForecastObject:

    def __init__(self, arquitetura: str):
        self.modelo = self.gerar_forecasts_no_problema_escolhido(arquitetura)

    # Standardization / Normalization
    @staticmethod
    def prod_normalizacao_stock_dfs(df_target: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
        df_copy = df_target.copy()


        df_index = df_copy.index
        df_copy.replace(-np.inf, 0, inplace=True)
        df_copy.replace(np.inf, 0, inplace=True)

        df_labels = df_copy.columns

        df_copy = scaler.transform(df_copy)
        df_copy = pd.DataFrame(data=df_copy, columns=df_labels, index=df_index)
        
        df_copy.dropna(inplace=True)

        df_copy[df_copy.select_dtypes(np.float64).columns] = df_copy.select_dtypes(np.float64).astype(np.float32)

        return df_copy

    def gerar_forecasts_no_problema_escolhido(self, arquitetura: str):
        df = pd.read_csv(f"{BASE_DIR}dados/pre_processado/salvador.csv")
        regressor: Model = keras.models.load_model(LSTM_MODELS_DIR+arquitetura+"/modelo_treinado.keras")
        scaler: MinMaxScaler = joblib.load(LSTM_MODELS_DIR+arquitetura+f"scaler.gz")

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
        df.drop(labels=["Data_Horario"], axis=1, inplace=True)

        # Normalização 
        df_normalizado = LSTMForecastObject.prod_normalizacao_stock_dfs(df_target=df)

        # Preparação dos Dados

        # Train-Validation-Test split
        numero_amostras_treinamento = int(0.6 * df.shape[0])  # 60%
        numero_amostras_validacao = int(0.2 * df.shape[0])  # 20%
        numero_amostras_teste = (
            df.shape[0] - numero_amostras_treinamento - numero_amostras_validacao
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
        n_features_measurements = len(input_measurements)
        n_features_forecast = len(input_features_forecast)

    
        input_test, output_test = split_sequence(
            HashableDataFrame(df_teste),
            look_back=LOOK_BACK,
            forecast_horizon=forecast_horizon,
            padding_between_lookback_forecast=PADDING_BETWEEN_LOOK_BACK_FORECAST,
            labels_input_measurements=tuple(input_measurements),
            labels_input_forecasts=tuple(input_features_forecast),
            labels_output=tuple(output_features),
        )

        # Gera previsões para cada step horário definido no problema
        df_last = df[
            numero_amostras_treinamento + numero_amostras_validacao :
        ]
        series_data_last = pd.to_datetime(df_last["Data_Horario"], infer_datetime_format=True)
        indices_horarios_5am = []
        diff_between_dataset_and_data = len(output_test) - len(series_data_last)
        for index, data_series_item in enumerate(series_data_last):
            if data_series_item.hour == 5 and index+diff_between_dataset_and_data >= 0:
                indices_horarios_5am.append(index+diff_between_dataset_and_data)
        return LSTMModel(regressor=regressor, mae=[0], mse=[0], r2=[0], y_pred=[0])

        #return LSTMModel(regressor=regressor, mae=mae, mse=mse, r2=r2, y_pred=y_pred)

    def get_forecast_simple_model(self, data_especifica: Optional[datetime.date]):
        # Dataframes futuros para a previsão
        N_DATAFRAMES_TARGET_MEASURE = 5

        # INDICADORES
        INDICADORES = ["EMA", "VWMA"]

        # Features de Interesse:
        features = ["date", "low", "open", "high", "volume", "close"]

        # Config Indicadores Tecnicos
        df_features = juntar_dataframes_stock(df_principal=self.df_primario, dfs_secundarios=[], features=features, indicadores=INDICADORES, indicadores_tecnicos_config=self.indicadores_config)
        df_features.reset_index(inplace=True)

        df_features["date"] = pd.to_datetime(df_features["date"])

        # Se tiver data especifica, pega os dados até o dia especificado
        if data_especifica is not None:
            data_especifica_datetime=datetime(year=data_especifica.year, month=data_especifica.month, day=data_especifica.day)
            df_features = df_features[df_features["date"] <= data_especifica_datetime]

        
        datas = df_features["date"]

        # Drop NA
        df_features.dropna(axis=0, inplace=True)

        # Drop date
        df_features.drop(labels=["date"], axis=1, inplace=True)

        # Normalização
        df_features_normalized = prod_scaling_stock_dfs(df_features=df_features, numero_stocks_secundarios=0, scaler=self.scaler)

        # Adaptacao do Dataset para formato esperado por predict
        df_features_last_that_matters = np.array(df_features_normalized.tail(self.kerasmodel.input_shape[1])).reshape(-1, self.kerasmodel.input_shape[1], self.kerasmodel.input_shape[2])

        # Previsão
        predicao = self.kerasmodel.predict(df_features_last_that_matters)
        return predicao[0][0]

    def get_forecast_historic(self):
        # INDICADORES
        INDICADORES = ["EMA", "VWMA"]

        # Features de Interesse:
        features = ["date", "low", "open", "high", "volume", "close"]

        # Config Indicadores Tecnicos
        df_features = juntar_dataframes_stock(df_principal=self.df_primario, dfs_secundarios=[], features=features, indicadores=INDICADORES, indicadores_tecnicos_config=self.indicadores_config)
        df_features.reset_index(inplace=True)

        # Drop date
        datas = df_features["date"]
        df_features.drop(labels=["date"], axis=1, inplace=True)

        df_features.reset_index(drop=True, inplace=True)

        # Drop NA
        df_features.dropna(axis=0, inplace=True)

        # Normalização
        df_features_normalized = prod_scaling_stock_dfs(df_features=df_features, numero_stocks_secundarios=0, scaler=self.scaler)

        # Criação de Datasets
        sampling_rate = 1
        sequence_length = 20
        delay_output = 0 # Delay de 5 dias pro output em relação ao fim do input, já calculado anteriormente
        delay = sampling_rate * (sequence_length + delay_output - 1)

        dados_features = df_features_normalized.to_numpy(copy=True)

        prediction_dataset = keras.utils.timeseries_dataset_from_array(
            data=dados_features,
            targets=None,
            sampling_rate=sampling_rate,
            sequence_length=sequence_length,
            shuffle=False,
            batch_size=1024,
        )

        # Adaptacao do Dataset para formato esperado por predict
        #df_features_that_matters = np.array(df_features_normalized_sem_target).reshape(-1, self.kerasmodel.input_shape[1], self.kerasmodel.input_shape[2])
        # Previsão 
        predicao = self.kerasmodel.predict(prediction_dataset)
        predicao_df = pd.DataFrame(pd.Series(predicao.reshape(-1)), columns=["predicao"])
        predicao_targets_df = pd.DataFrame(datas.copy())
        predicao_targets_df.set_index("date", inplace=True)

        predicao_df["date"] = datas[-predicao_df.shape[0]:].array
        predicao_df.set_index("date", inplace=True)

        predicao_targets_df["predicao"] = predicao_df["predicao"]
        predicao_targets_df.dropna(axis=0, inplace=True)
        return predicao_targets_df