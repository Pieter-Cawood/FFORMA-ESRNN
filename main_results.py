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

def smape_loss(y_true, y_pred):
    epsilon = 0.35
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape

def record_comination_owas(comb_owa, total_owa):
    ret_owa = total_owa
    if total_owa is None:
        ret_owa = comb_owa
    else:
        ret_owa = np.append(ret_owa, comb_owa)
    return ret_owa

def forecast_error_reduction_mechanism(y_hat, y_hat_base_models, threshold=0.4):
    averaging_preds = y_hat_base_models.sum(axis=1) / y_hat_base_models.shape[1]


def run(df_info, df_train_data, df_pred_data,
        seasonality,
        k_folds=10, n_runs=5, optimizing_runs=0, combination_type='FFORMA', hyper_search_run=False):
    print(f"results/{combination_type}_{seasonality[0]}.pd")
    try:
        df_results = pd.read_pickle(f"results/{combination_type}_{seasonality[0]}.pd")        
        print(df_results)
    except:
        print(f"No results")
    print(45*"-")

if __name__ == '__main__':
    for seasonality in ['Hourly','Daily','Weekly','Monthly','Yearly','Quarterly']:
        # seasonality = 'Daily'
        X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(seasonality, 
                                                                 'data', 
                                                                 'forecasts', 
                                                                 load_existing_dataframes=True)
        for combination_type in ['nbeats',
                                 'FFORMA',
                                 'FFORMS',
                                 'Model Averaging',
                                 'Neural Averaging 2',
                                 'Neural Stacking',
                                 'Deep FFORMA_VGG',
                                 'Deep FFORMA_RESNET'][-1:]:
            run(df_info=X_test_df,
                df_train_data=y_train_df,
                df_pred_data=y_test_df,
                seasonality=seasonality,
                optimizing_runs=0,
                combination_type=combination_type,
                n_runs=5,
                k_folds=10,
                hyper_search_run=False)
