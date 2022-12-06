from functools import lru_cache
import gc
from hashlib import sha256
import logging
import os
import pickle
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from pandas.util import hash_pandas_object
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = "/home/b-rbmp-ideapad/Documents/GitHub/ml-solar-forecaster/"
GTB_MODELS_DIR = BASE_DIR + "desenvolvimento_modelos/sklearn_models/models/gradienttreeboosting/"

class HashableDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)

from dataclasses import dataclass

@dataclass(order=True)
class RegressorStepModel:
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

 # Configuracao Logging
def create_logger(debug_mode: bool):
    logging_level: int
    handler: logging.FileHandler
    if not debug_mode:
        handler = logging.FileHandler('gtb.log')
        logging_level = logging.INFO
    else:
        handler = logging.FileHandler('gtb.debug')
        logging_level = logging.DEBUG

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger('GTBExploracaoLogger')
    logger.setLevel(logging_level)
    logger.addHandler(handler)
    return logger


class GTBSolarRegressor:
    def __init__(self, target_local: str = "salvador", train_test_split_ratio: float = 0.2, custom_input_features: List[str] = [], in_n_measures: int = 168, out_n_measures: int = 24, n_estimators: int = 1000, loss = 'squared_error', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None, modo_otimizacao=False):
        self.target_local = target_local
        self.in_n_measures = in_n_measures
        self.out_n_measures = out_n_measures
        self.train_test_split = train_test_split_ratio
        self.n_estimators = n_estimators
        self.rodar_instancia_treinamento(target_local=target_local, custom_input_features=custom_input_features, train_test_split_ratio=train_test_split_ratio, in_n_measures=in_n_measures, out_n_measures=out_n_measures, n_estimators=n_estimators, loss=loss, learning_rate=learning_rate, subsample=subsample, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, modo_otimizacao=modo_otimizacao)

    # Funcao que roda uma instãncia de treinamento e retorna o scaler,
    def rodar_instancia_treinamento(self, target_local: str = "salvador", train_test_split_ratio: float = 0.2, custom_input_features: List[str] = [], in_n_measures: int = 168, out_n_measures: int = 24, n_estimators: int = 1000, loss = 'squared_error', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None, modo_otimizacao=False):

        # Download dos dados de treinamento
        df_target = pd.read_csv(f"{BASE_DIR}dados/pre_processado/{target_local}.csv")
        df_target[df_target.select_dtypes(np.float64).columns] = df_target.select_dtypes(np.float64).astype(np.float16)
        # Remove coluna de Index
        df_target.drop(df_target.columns[df_target.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

        # Define o valor de max_features 
        max_features = None

        # Preparação dos Dados

        # Transformações para adicionar colunas de hora, dia, mês e ano
        series_data = pd.to_datetime(df_target["Data_Horario"], infer_datetime_format=True)

        # Criando uma Feature Cíclica representado a posição do Momento no Ano
        s = 24 * 60 * 60 # Segundos no ano 
        year = (365.25) * s
        df_target["timestamp"] = series_data.values.astype('int64') // 10**9
        df_target["ano_cos"] = [np.cos((x) * (2 * np.pi / year)) for x in df_target["timestamp"]]
        df_target["ano_sin"] = [np.sin((x) * (2 * np.pi / year)) for x in df_target["timestamp"]]
        df_target.drop(labels=["timestamp"], axis=1, inplace=True)

        # Criando uma Feature Cíclica representado a posição da Hora no Dia
        df_target["hora_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in series_data.dt.hour]
        df_target["hora_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in series_data.dt.hour]
        df_target.drop(labels=["Data_Horario"], axis=1, inplace=True)

        # Normalização 
        label_target = "IRRADIÂNCIA"
        df_target_normalized, scaler = GTBSolarRegressor.normalizacao_stock_dfs(df_target=df_target, label_target=label_target)

        # Features Entrada
        if len(custom_input_features) > 0:
            input_features_forecast = custom_input_features
        else:
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
        output_features = ["IRRADIÂNCIA"]
        
        del df_target
        gc.collect()

        data_multioutput_supervised = gerar_dados_problema_supervisionado(forecast_data=HashableDataFrame(df_target_normalized[input_features_forecast]), measurements_data=HashableDataFrame(df_target_normalized[input_measurements]), output_data=HashableDataFrame(df_target_normalized.iloc[in_n_measures:][output_features]), n_in=in_n_measures, n_out=out_n_measures)

        target_output_after_treatment = "IRRADIÂNCIA TARGET"
        range_models = range(0, len(data_multioutput_supervised)) if not modo_otimizacao else range(0, len(data_multioutput_supervised), math.floor(0.30*len(data_multioutput_supervised)))
        range_models = range(19, 24)
        for i in range_models:
            step = i + 1
            df_data = data_multioutput_supervised[i]
            X_train, X_test, Y_train, Y_test = train_test_split(df_data.drop(labels=[target_output_after_treatment], axis=1, inplace=False), df_data[target_output_after_treatment], test_size=train_test_split_ratio, random_state=43, shuffle=False)
            del df_data
            gc.collect()

            regressor = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, subsample=subsample, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, n_estimators=n_estimators, random_state=43, validation_fraction=0.25, n_iter_no_change=50, verbose=1, tol=1e-4)
            regressor.fit(X_train, Y_train)

            Y_pred = regressor.predict(X_test)
            mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
            mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
            r2 = r2_score(y_true=Y_test, y_pred=Y_pred)

           #LOGGER.info(f"STEP: {step}h | mae: {mae} | mse: {mse}")

            regressor_step_model = RegressorStepModel(step=step, regressor=regressor, mae=mae, mse=mse, r2=r2, scaler=scaler)

            filename = f'{step}h.gtbmodel'
            pickle.dump(regressor_step_model, open(GTB_MODELS_DIR + filename, 'wb'))
            
            del regressor, regressor_step_model
            gc.collect()
    

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

        df_copy[df_copy.select_dtypes(np.float64).columns] = df_copy.select_dtypes(np.float64).astype(np.float16)

        return df_copy, scaler

def load_all_gtb_models() -> List[RegressorStepModel]:
    models: List[RegressorStepModel] = []
    filenames = [name for name in os.listdir(GTB_MODELS_DIR) if os.path.isfile(os.path.join(GTB_MODELS_DIR, name))]
    for f_name in filenames:
        file = os.path.join(GTB_MODELS_DIR, f_name)  # full path
        model: RegressorStepModel = pickle.load(open(file, 'rb'))
        model.__post_init__()
        models.append(model)
    return models


LOGGER = create_logger(debug_mode=False)

# features = [
#     "ano_cos",
#     "ano_sin",
#     "hora_cos",
#     "hora_sin",
# ]
# features_a_adicionar = [
#     "WS10M",
#     "RH2M",
#     "WD50M",
#     "PRECTOTCORR",
#     "T2M",
#     "WD10M",
#     "WS50M",
#     "PSC",
#     "T2MWET",
#     "T2MDEW",
# ]

# for custom_input_feature in features_a_adicionar:
#     features_nova = features.copy()
#     features_nova.append(custom_input_feature)
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=1000, custom_input_features=features_nova, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA: {features_nova}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2

#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")

                
#     del treinamento, models
#     gc.collect()

features_final = [
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

#Hyperparameter Search
n_estimators_search = [10, 100, 200, 300, 1000, 2000, 5000]
loss_search = ['squared_error', 'huber', 'quantile']
learning_rate_search = [0.01, 0.05, 0.1, 0.5, 1]
subsample_search = [0.2, 0.5, 0.8, 1]
criterion_search = ['friedman_mse', 'squared_error', 'mse']
max_depth_search = [2, 3, 5, 8, 12, 14]
min_samples_split_search = [2, 3, 5, 7]
min_samples_leaf_search = [1, 2, 3, 5, 7]
max_features_search = ["sqrt", "log2", None]

# for n_estimators in n_estimators_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators, custom_input_features=features_final, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA n_estimators={n_estimators}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

n_estimators_final = 1000
# for loss in loss_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA loss={loss}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()


loss_final = "squared_error"
# for learning_rate in learning_rate_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA learning_rate={learning_rate}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

learning_rate_final = 0.1
# for subsample in subsample_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA subsample={subsample}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

subsample_final = 1.0
# for criterion in criterion_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample_final, criterion=criterion, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA criterion={criterion}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

criterion_final = "friedman_mse"
# for max_depth in max_depth_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample_final, criterion=criterion_final, max_depth=max_depth, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA max_depth={max_depth}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

max_depth_final = 3
# for min_samples_split in min_samples_split_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample_final, criterion=criterion_final, max_depth=max_depth_final, min_samples_split=min_samples_split, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA min_samples_split={min_samples_split}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

min_samples_split_final = 2
# for min_samples_leaf in min_samples_leaf_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample_final, criterion=criterion_final, max_depth=max_depth_final, min_samples_split=min_samples_split_final, min_samples_leaf=min_samples_leaf, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA min_samples_leaf={min_samples_leaf}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

min_samples_leaf_final = 1
# for max_features in max_features_search:
#     treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample_final, criterion=criterion_final, max_depth=max_depth_final, min_samples_split=min_samples_split_final, min_samples_leaf=min_samples_leaf_final, max_features=max_features, modo_otimizacao=True)
#     LOGGER.info(f"RESULTADOS PARA max_features={max_features}")
#     # Mostrar Resultados
#     models = sorted(load_all_gtb_models())

#     mae_total = 0.00
#     mse_total = 0.00
#     r2_total = 0.00
#     count = 0
#     for model in models:
#         mae = model.mae
#         mse = model.mse
#         r2 = model.r2
#         mae_total += mae
#         mse_total += mse
#         r2_total += r2
#         count += 1
#         LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

#     mae_global = mae_total/count
#     mse_global = mse_total/count
#     r2_global = r2_total/count
#     LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")
#     del treinamento, models
#     gc.collect()

max_features_final = None
treinamento = GTBSolarRegressor(target_local="salvador", train_test_split_ratio=0.2, in_n_measures=24, out_n_measures=24, n_estimators=n_estimators_final, custom_input_features=features_final, loss=loss_final, learning_rate=learning_rate_final, subsample=subsample_final, criterion=criterion_final, max_depth=max_depth_final, min_samples_split=min_samples_split_final, min_samples_leaf=min_samples_leaf_final, max_features=max_features_final)

# Mostrar Resultados
# models = sorted(load_all_gtb_models())

# mae_total = 0.00
# mse_total = 0.00
# r2_total = 0.00
# count = 0
# for model in models:
#     mae = model.mae
#     mse = model.mse
#     r2 = model.r2
#     mae_total += mae
#     mse_total += mse
#     r2_total += r2
#     count += 1
#     LOGGER.info(f"STEP: {model.step}h | mae: {mae} | mse: {mse} | R2: {r2}")

# mae_global = mae_total/count
# mse_global = mse_total/count
# r2_global = r2_total/count
# LOGGER.info(f"TOTAL => mae: {mae_global} | mse: {mse_global} | R2: {r2_global}")