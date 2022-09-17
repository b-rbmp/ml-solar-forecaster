# Imports
from base64 import encode
from functools import lru_cache
from hashlib import sha256
from pandas.util import hash_pandas_object
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from typing import Callable, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

BASE_DIR = "/home/b-rbmp-ideapad/Documents/GitHub/ml-solar-forecaster/"
LSTM_MODELS_DIR = BASE_DIR + "desenvolvimento_modelos/neural_networks/models"

@lru_cache(64)
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
    X = np.concatenate(X_measurements, X_forecasts, axis=1)
    return X, np.array(y) # Testar


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
        handler = logging.FileHandler('lstm.log')
        logging_level = logging.INFO
    else:
        handler = logging.FileHandler('lstm.debug')
        logging_level = logging.DEBUG

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger('LstmNasaInmetLogger')
    logger.setLevel(logging_level)
    logger.addHandler(handler)

    return logger

def vanilla_lstm_model(
    look_back: int, n_features: int, forecast_range: int
):
    inputs = keras.Input(shape=(look_back, n_features))
    lstm1 = layers.LSTM(200)(inputs)
    dropout1 = layers.Dropout(rate=0.15)(lstm1)
    dense1 = layers.Dense(100, activation="relu")(dropout1)
    dropout2 = layers.Dropout(rate=0.15)(dense1)
    outputs = layers.Dense(forecast_range)(dropout2)
    model = Model(inputs, outputs)
    return model


def encoder_decoder_lstm_model(
    look_back: int, n_features: int, forecast_range: int
):
    inputs = keras.Input(shape=(look_back, n_features))
    lstm1 = layers.LSTM(128)(inputs)
    dropout1 = layers.Dropout(rate=0.15)(lstm1)
    repeat_vector = layers.RepeatVector(forecast_range)(dropout1)
    lstm2 = layers.LSTM(128, return_sequences=True)(repeat_vector)
    dropout2 = layers.Dropout(rate=0.15)(lstm2)
    timedistributed_dense1 = layers.TimeDistributed(layers.Dense(64))(dropout2)
    timedistributed_dropout1 = layers.TimeDistributed(layers.Dropout(rate=0.15))(timedistributed_dense1)
    timedistributed_dense2 = layers.TimeDistributed(layers.Dense(1))(timedistributed_dropout1)
    outputs = timedistributed_dense2
    model = Model(inputs, outputs)
    return model

    
def cnn_lstm_encoder_decoder_model(
    look_back: int, n_features: int, forecast_range: int
):
    inputs = keras.Input(shape=(look_back, n_features))
    conv1 = layers.Conv1D(filters=64, kernel_size=9, activation='relu')(inputs)
    conv2 = layers.Conv1D(filters=64, kernel_size=11, activation='relu')(conv1)
    maxpolling1 = layers.MaxPooling1D(pool_size=2)(conv2)
    dropout1 = layers.Dropout(rate=0.15)(maxpolling1)
    flatten1 = layers.Flatten()(dropout1)
    repeat_vector = layers.RepeatVector(forecast_range)(flatten1)
    lstm1 = layers.LSTM(128, return_sequences=True)(repeat_vector)
    dropout2 = layers.Dropout(rate=0.15)(lstm1)
    timedistributed_dense1 = layers.TimeDistributed(layers.Dense(64, activation='relu'))(dropout2)
    timedistributed_dropout1 = layers.TimeDistributed(layers.Dropout(rate=0.15))(timedistributed_dense1)
    timedistributed_dense2 = layers.TimeDistributed(layers.Dense(1))(timedistributed_dropout1)
    outputs = timedistributed_dense2
    model = Model(inputs, outputs)
    return model



class NeuralTrainingModel:
    """docstring for NeuralTrainingModel."""

    def __init__(
        self,
        train_test_split: List[int] = [60, 20, 20],
        in_n_measures: int = 168,
        out_n_measures: int = 24,
        sampling_rate: int = 1,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        model_function: Callable = vanilla_lstm_model,
    ):
        self.train_test_split = train_test_split
        self.in_n_measures = in_n_measures
        self.out_n_measures = out_n_measures
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_function = model_function

        self.mae_teste_kj, self.loss_values, self.val_loss_values = self.rodar_instancia_treinamento()

    # Plot do MAE de validação x MAE de treino conforme o treinamento avançava
    def plot_training_validation_loss(self):
        epochs = range(1, len(self.loss_values) + 1)
        plt.plot(epochs, self.loss_values, "bo", label="Loss de Treinamento")
        plt.plot(epochs, self.val_loss_values, "b", label="Loss de Validação")
        plt.title("Loss de Treinamento e Validação")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    # Scatter plot de Previsões x Target
    def plot_scatter_previsoes_targets(self):
        plt.scatter(
            x=self.df_previsoes_targets["previsoes"],
            y=self.df_previsoes_targets["targets"],
        )

    # Funcao que roda uma instãncia de treinamento e retorna o scaler, mae, mae_persistencia, distribuição de erros e a historia
    def rodar_instancia_treinamento(self):
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
        label_target = "IRRADIÂNCIA"
        df_normalizado, scaler = NeuralTrainingModel.normalizacao_stock_dfs(df_target=df, label_target=label_target)

        # Preparação dos Dados

        # Train-Validation-Test split
        numero_amostras_treinamento = int(0.6 * df.shape[0])  # 60%
        numero_amostras_validacao = int(0.2 * df.shape[0])  # 20%
        numero_amostras_teste = (
            df.shape[0] - numero_amostras_treinamento - numero_amostras_validacao
        )  # % 20%

        df_train = df_normalizado[0:numero_amostras_treinamento]
        df_validacao = df_normalizado[
            numero_amostras_treinamento : numero_amostras_treinamento
            + numero_amostras_validacao
        ]
        df_teste = df_normalizado[
            numero_amostras_treinamento + numero_amostras_validacao :
        ]

        # Features Entrada
        input_features_forecast = [
            "T2M",
            "T2MDEW",
            "T2MWET",
            "RH2M",
            "PRECTOTCORR",
            "WD10M",
            "WS10M",
            "WS50M",
            "WD50M",
            "PSC",
            "ano_cos",
            "ano_sin",
            "hora_cos",
            "hora_sin",
        ]

        input_measurements = [
            "IRRADIÂNCIA"
        ]

        # Features Saida
        output_features = [
            "IRRADIÂNCIA"
        ]
        

        # Persistência
        persistencia_validacao = NeuralTrainingModel.verificacao_persistencia_mae(
            df_validacao, scaler=scaler, output_label=output_features[0]
        )
        persistencia_teste = NeuralTrainingModel.verificacao_persistencia_mae(
            df_teste, scaler=scaler, output_label=output_features[0]
        )

        LOOK_BACK = self.in_n_measures
        FORECAST_RANGE = self.out_n_measures
        PADDING_BETWEEN_LOOK_BACK_FORECAST = 1
        n_features = len(input_features_forecast) + len(input_measurements)

        input_train, output_train = split_sequence(
            HashableDataFrame(df_train),
            look_back=LOOK_BACK,
            forecast_horizon=FORECAST_RANGE,
            padding_between_lookback_forecast=PADDING_BETWEEN_LOOK_BACK_FORECAST,
            labels_input_measurements=tuple(input_measurements),
            labels_input_forecasts=tuple(input_features_forecast),
            labels_output=tuple(output_features),
        )
        input_validation, output_validation = split_sequence(
            HashableDataFrame(df_validacao),
            look_back=LOOK_BACK,
            forecast_horizon=FORECAST_RANGE,
            padding_between_lookback_forecast=PADDING_BETWEEN_LOOK_BACK_FORECAST,
            labels_input_measurements=tuple(input_measurements),
            labels_input_forecasts=tuple(input_features_forecast),
            labels_output=tuple(output_features),
        )
        input_test, output_test = split_sequence(
            HashableDataFrame(df_teste),
            look_back=LOOK_BACK,
            forecast_horizon=FORECAST_RANGE,
            padding_between_lookback_forecast=PADDING_BETWEEN_LOOK_BACK_FORECAST,
            labels_input_measurements=tuple(input_measurements),
            labels_input_forecasts=tuple(input_features_forecast),
            labels_output=tuple(output_features),
        )

        modelo: Model = self.model_function(LOOK_BACK, n_features, FORECAST_RANGE)
        callbacks = [
                keras.callbacks.ModelCheckpoint("modelo_lstm.keras", save_best_only=True),
                keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=50,
                            restore_best_weights=True,
                ),
            ]
            
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])

        # Test run para o caso a)
        history = modelo.fit(input_train,
            output_train,
            batch_size=self.batch_size,
            epochs=200,
            validation_data=(input_validation, output_validation),
            callbacks=callbacks)


        MAE_teste = modelo.evaluate(x=input_test, y=output_test, batch_size=self.batch_size)[1]

        mae_transform_scaler_shape_in = np.zeros(shape=[1, 13])
        mae_transform_scaler_shape_in[0][2] = MAE_teste
        MAE_teste_kj = scaler.inverse_transform(mae_transform_scaler_shape_in)
        MAE_teste_kj = MAE_teste_kj[0][2]

        history_dict = history.history
        loss_values = history_dict["loss"][5:]
        val_loss_values = history_dict["val_loss"][5:]
        #forecast_teste = modelo.predict(x=input_test, batch_size=None)

        return MAE_teste_kj, loss_values, val_loss_values

    # Standardization / Normalization
    @staticmethod
    def normalizacao_stock_dfs(df_target: pd.DataFrame, normalizacao_or_standardizacao: str = "normalizacao", label_target: str | None = "IRRADIÂNCIA") -> Tuple[pd.DataFrame, MinMaxScaler | StandardScaler | None]:
        scaler: MinMaxScaler | StandardScaler | None
        if normalizacao_or_standardizacao == 'normalizacao':
            scaler = MinMaxScaler()
        elif normalizacao_or_standardizacao == 'standardizacao':
            scaler = StandardScaler()
        
        
        df_copy = df_target.copy()

        # Não normaliza o target
        if label_target is not None:
            df_target = df_copy[label_target]
        df_index = df_copy.index
        df_copy.replace(-np.inf, 0, inplace=True)
        df_copy.replace(np.inf, 0, inplace=True)
        df_copy.drop(labels=[label_target], axis=1, inplace=True)
        df_labels = df_copy.columns

        df_copy = scaler.fit_transform(df_copy)
        df_copy = pd.DataFrame(data=df_copy, columns=df_labels, index=df_index)
        
        if label_target is not None:
            df_copy[label_target] = df_target
        df_copy.dropna(inplace=True)

        df_copy[df_copy.select_dtypes(np.float64).columns] = df_copy.select_dtypes(np.float64).astype(np.float32)

        return df_copy, scaler

    @staticmethod
    # Calculo MAE previsões
    # Verificação de Modelo calculando o MAE
    def calculo_mae(previsoes, targets):
        total_abs_err = 0
        amostras_vistas = 0
        for i in range(len(previsoes)):
            previsao = previsoes[i]
            target = targets[i].reshape(-1)
            total_abs_err += np.sum(np.abs(previsao-target))
            amostras_vistas += len(previsao)

        return total_abs_err / amostras_vistas


    @staticmethod
    def verificacao_persistencia_mae(
        df: pd.DataFrame, scaler: MinMaxScaler, output_label: str
    ):
        total_abs_err = 0
        amostras_vistas = 0
        df_copy = df.copy()
        index = df_copy.index
        columns = df_copy.columns
        unscaled_data = scaler.inverse_transform(df_copy)
        df_copy = pd.DataFrame(unscaled_data, index=index, columns=columns)
        df_copy["previsao_persistencia"] = df_copy[output_label].shift(24)
        df_copy["erro_persistencia_mod"] = abs(
            df_copy["previsao_persistencia"] - df_copy[output_label]
        )
        df_copy_without_nan = df_copy.dropna()
        mae = df_copy["erro_persistencia_mod"].mean(axis=0, skipna=True)

        df_series_targets = df_copy_without_nan[["previsao_persistencia", output_label]]
        return mae, df_series_targets

    @staticmethod
    def evaluate_forecast(y_test_inverse, yhat_inverse):
        mse_ = tf.keras.losses.MeanSquaredError()
        mae_ = tf.keras.losses.MeanAbsoluteError()
        mape_ = tf.keras.losses.MeanAbsolutePercentageError() 
        mae = mae_(y_test_inverse,yhat_inverse)
        mse = mse_(y_test_inverse,yhat_inverse)
        mape = mape_(y_test_inverse,yhat_inverse)
        return mae, mse, mape



LOGGER = create_logger(debug_mode=False)



LOGGER.info(f"Iniciando Treinamentos")
treinamento_object_vanilla_label = NeuralTrainingModel(model_function=vanilla_lstm_model)
LOGGER.info(f'VANILLA: {treinamento_object_vanilla_label.mae_teste_kj}')
treinamento_object_encoder_decoder_label = NeuralTrainingModel(model_function=encoder_decoder_lstm_model)
LOGGER.info(f'ENCODER_DECODER: {treinamento_object_encoder_decoder_label.mae_teste_kj}')
treinamento_object_cnn_lstm_label = NeuralTrainingModel(model_function=cnn_lstm_encoder_decoder_model)
LOGGER.info(f'CNN_LSTM: {treinamento_object_cnn_lstm_label.mae_teste_kj}')
print("----")

print("FIM")


