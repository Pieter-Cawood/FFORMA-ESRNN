from fforma import FFORMA
from fforma.optimization import *
from neuralaverage.mlp import *
from stacking.mlp import *
from utils.data import *
import pandas as pd
import numpy as np
from utils.analysis import evaluate_prediction_owa
from configs.configs import FFORMA_CONFIGS, FEATURE_CONFIGS, NEURALSTACK_CONFIGS, NEURALAVERAGE_CONFIGS
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import locally_linear_embedding
from sklearn.preprocessing import StandardScaler, RobustScaler
from lightgbm import LGBMRegressor
import lightgbm as lgb
from nbeats_keras.model import NBeatsNet
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanAbsolutePercentageError
import keras.backend as K
import seaborn as sns

def run(df_info, df_train_data, df_pred_data,
        seasonality):
    owa_esrnn_df = df_pred_data['OWA_mdl_ESRNN']
    owa_arima_df = df_pred_data['OWA_mdl_ARIMA']
    owa_comb_df = df_pred_data['OWA_mdl_Comb']
    owa_damped_df = df_pred_data['OWA_mdl_Damped']
    owa_theta_df = df_pred_data['OWA_mdl_Theta']

    pred_esrnn_df = df_pred_data['mdl_ESRNN']
    pred_arima_df = df_pred_data['mdl_ARIMA']
    pred_comb_df = df_pred_data['mdl_Comb']
    pred_damped_df = df_pred_data['mdl_Damped']
    pred_theta_df = df_pred_data['mdl_Theta']

    print('ESRNN Average OWA: {}'.format(np.average(owa_esrnn_df.values)))
    print('ARIMA Average OWA: {}'.format(np.average(owa_arima_df.values)))
    print('Comb Average OWA: {}'.format(np.average(owa_comb_df.values)))
    print('Damped Average OWA: {}'.format(np.average(owa_damped_df.values)))
    print('Theta Average OWA: {}'.format(np.average(owa_theta_df.values)))

    print('ESRNN Median OWA: {}'.format(np.median(owa_esrnn_df.values)))
    print('ARIMA Median OWA: {}'.format(np.median(owa_arima_df.values)))
    print('Comb Median OWA: {}'.format(np.median(owa_comb_df.values)))
    print('Damped Median OWA: {}'.format(np.median(owa_damped_df.values)))
    print('Theta Median OWA: {}'.format(np.median(owa_theta_df.values)))

    print('ESRNN R2 loss: {}'.format(r2_score(df_pred_data.y, pred_esrnn_df.values)))
    print('ARIMA R2 loss: {}'.format(r2_score(df_pred_data.y, pred_arima_df.values)))
    print('Comb R2 loss: {}'.format(r2_score(df_pred_data.y, pred_comb_df.values)))
    print('Damped R2 loss: {}'.format(r2_score(df_pred_data.y, pred_damped_df.values)))
    print('Theta R2 loss: {}'.format(r2_score(df_pred_data.y, pred_theta_df.values)))



if __name__ == '__main__':
    seasonality = 'Hourly'
    X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(seasonality, 'data', 'forecasts', load_existing_dataframes=True)
    run(df_info=X_test_df,
        df_train_data=y_train_df,
        df_pred_data=y_test_df,
        seasonality=seasonality
        )
