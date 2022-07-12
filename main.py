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
    overall_combination_loss = 0.0
    overall_combination_loss_median = 0.0
    overall_combination_loss_r2 = 0.0
    overall_esrnn_loss = 0.0

    total_combination_owa = None

    for run_num in range(1 if hyper_search_run else n_runs):
        combination_run_loss = 0.0
        combination_run_loss_median = 0.0
        combination_run_loss_r2 = 0.0
        esrnn_run_loss = 0.0

        kfoldings = make_kfolds(df_info, df_pred_data, k_folds, seed=run_num)
        #Changed this to only do the first fold of the first run as per paper for ypers
        for test_fold_num in range(1 if hyper_search_run else k_folds):
            train_set, test_set = train_test_split(kfoldings, test_fold_num)

            base_model_names = []
            for base_model in train_set.columns[train_set.columns.str.startswith('mdl_')].tolist():
                if base_model != 'mdl_naive2':
                    base_model_names.append(base_model)
            if (combination_type == 'nbeats'):
                train_feats = train_set.copy()
                train_feats = train_feats.set_index('unique_id')
                train_feats = train_feats.filter(regex='^mf_', axis=1)
                train_feats = train_feats[FEATURE_CONFIGS[seasonality]]
                train_feats = train_feats.values
                train_feats = np.reshape(train_feats, (train_set['unique_id'].nunique(),
                                                        seas_dict[seasonality]['output_size'],
                                                        train_feats.shape[1]))
                test_feats = test_set.copy()
                test_feats = test_feats.set_index('unique_id')
                test_feats = test_feats.filter(regex='^mf_', axis=1)
                test_feats = test_feats[FEATURE_CONFIGS[seasonality]]
                test_feats = test_feats.values
                test_feats = np.reshape(test_feats, (test_set['unique_id'].nunique(),
                                                      seas_dict[seasonality]['output_size'],
                                                      test_feats.shape[1]))

            elif (combination_type != 'Neural Stacking'):
                train_feats = train_set.copy()
                train_feats = train_feats.drop_duplicates('unique_id').set_index('unique_id')
                train_feats = train_feats.filter(regex='^mf_', axis=1)
          #  temp = train_set.copy()
         #   temp = temp.drop_duplicates('unique_id').set_index('unique_id')
          #  train_feats['cat'] = temp['category'].values
                test_feats = test_set.copy()
                test_feats = test_feats.drop_duplicates('unique_id').set_index('unique_id')
                test_feats = test_feats.filter(regex='^mf_', axis=1)
           # temp = test_set.copy()
          #  temp = temp.drop_duplicates('unique_id').set_index('unique_id')
          #  test_feats['cat'] = temp['category'].values
            else:
                train_feats = train_set.copy()
                train_feats = train_feats.set_index('unique_id')
                train_feats = train_feats.filter(regex='^mf_', axis=1)
                test_feats = test_set.copy()
                test_feats = test_feats.set_index('unique_id')
                test_feats = test_feats.filter(regex='^mf_', axis=1)

           # train_feats = train_feats[FEATURE_CONFIGS[seasonality]]
          #  test_feats = test_feats[FEATURE_CONFIGS[seasonality]]
            #TODO: select cols in list
            #exclude_list = ['mf_variation', 'mf_summation']#['mf_nperiods','mf_seasonal_period', 'mf_variation', 'mf_summation']

            # Compute training errors
            train_errors = train_set.copy()
            train_errors = train_errors.drop_duplicates('unique_id').set_index('unique_id')
            train_errors = train_errors.filter(regex='^OWA_',axis=1)
            train_errors.columns = train_errors.columns.str.lstrip('OWA_')


            if optimizing_runs > 0:
                parameters = get_optimal_params(train_errors, train_feats, test_feats, optimizing_runs)
                print(parameters.x)
                quit()
            else:
                if combination_type in ['FFORMA','FFORMS']:
                    parameters = FFORMA_CONFIGS[seasonality]

            y_hat_base_models_train = train_set[['unique_id', 'ds'] + base_model_names].set_index(['unique_id', 'ds'])
            y_hat_base_models_test = test_set[['unique_id','ds'] + base_model_names].set_index(['unique_id','ds'])

            if combination_type == 'nbeats':
              #  extra_model_train = y_hat_base_models_train['mdl_245'].values
                y_hat_base_models_train = y_hat_base_models_train['mdl_ESRNN'].values
                y_hat_base_models_train = np.reshape(y_hat_base_models_train, (train_set['unique_id'].nunique(),
                                                     seas_dict[seasonality]['output_size']))#,

             #   extra_model_train = np.reshape(extra_model_train, (train_set['unique_id'].nunique(),
             #                                                      seas_dict[seasonality]['output_size'], 1))
              #  extra_model_train = extra_model_train / np.expand_dims(y_hat_base_models_train, axis=-1)

                extra_model_train = None

              #  extra_model_test = y_hat_base_models_test['mdl_245'].values
                y_hat_base_models_test = y_hat_base_models_test['mdl_ESRNN'].values
                y_hat_base_models_test = np.reshape(y_hat_base_models_test, (test_set['unique_id'].nunique(),
                                                     seas_dict[seasonality]['output_size']))

         #       extra_model_test = np.reshape(extra_model_test, (test_set['unique_id'].nunique(),
           #                                                     seas_dict[seasonality]['output_size'], 1))

            #    extra_model_test = extra_model_test / np.expand_dims(y_hat_base_models_test, axis=-1)

            if combination_type in ['FFORMA','FFORMS']:
                fforma = FFORMA(params=parameters, verbose_eval=1)
                fforma.fit(errors=train_errors, holdout_feats=train_feats, feats=test_feats)
                fforma_preds = fforma.predict(y_hat_base_models_test, combination_type).reset_index()
                # FFORMA SCORE
                test_fforma_df = test_set.copy()
                test_fforma_df['y_hat'] = fforma_preds['fformx_prediction'].values
                predictions_df = test_fforma_df[['unique_id', 'y_hat', 'ds']]
                combination_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                                y_train_df=y_train_df,
                                                                y_test_df=test_fforma_df,
                                                                naive2_seasonality=seas_dict[seasonality]['seasonality'],
                                                                return_averages=False)
                total_combination_owa = record_comination_owas(combination_owa, total_combination_owa)
                combination_owa_median = np.median(combination_owa)
                combination_owa = np.mean(combination_owa)

            elif combination_type in ['Neural Averaging','Neural Averaging 2']:                
                if combination_type== 'Neural Averaging':
                    navg = ModelAveragingMLP(NEURALAVERAGE_CONFIGS[seasonality],train_feats.shape[1],train_errors.shape[1], style=combination_type)
                else:
                    navg = ModelAveragingMLP(NEURALAVERAGE_CONFIGS[seasonality+"_2"],train_feats.shape[1],train_errors.shape[1], style=combination_type)
                    transformer = StandardScaler().fit(train_feats) #RobustScaler(quantile_range=(1.0, 99.0)).fit(train_feats)
                    lower = transformer.mean_
                    upper = transformer.scale_
                    train_feats = ((train_feats-lower)/(upper)).fillna(0)
                    test_feats  = ((test_feats -lower)/(upper)).fillna(0)
                navg.fit(train_feats, train_errors)
                #print(navg.weights(train_feats))
                navg_preds = navg.predict(test_feats, y_hat_base_models_test).reset_index()
                # NAVG SCORE
                test_navg_df = test_set.copy()
                test_navg_df['y_hat'] = navg_preds['navg_prediction'].values
                predictions_df = test_navg_df[['unique_id', 'y_hat', 'ds']]
                combination_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                                y_train_df=y_train_df,
                                                                y_test_df=test_navg_df,
                                                                naive2_seasonality=seas_dict[seasonality]['seasonality'],
                                                                return_averages=False)
                total_combination_owa = record_comination_owas(combination_owa, total_combination_owa)
                combination_owa_median = np.median(combination_owa)
                combination_owa = np.mean(combination_owa)

            elif combination_type == 'Neural Stacking':
                autoscaler = StandardScaler(with_mean= False, with_std = False)
               # y_hat_base_models_train[y_hat_base_models_train.columns] = autoscaler.fit_transform(y_hat_base_models_train[y_hat_base_models_train.columns])
                train_features = pd.concat([train_feats.reset_index(),
                                            y_hat_base_models_train.reset_index()], axis=1).\
                    set_index(['unique_id']).drop(columns=['ds'])
             #   y_hat_base_models_test[y_hat_base_models_test.columns] = autoscaler.fit_transform(y_hat_base_models_test[y_hat_base_models_test.columns])
                test_features = pd.concat([test_feats.reset_index(),
                                           y_hat_base_models_test.reset_index()], axis=1).\
                    set_index(['unique_id']).drop(columns=['ds'])
                nstack = StackingMLP(NEURALSTACK_CONFIGS[seasonality],
                                     train_features.shape[1])
                nstack.fit(train_features, train_set[['y']])
                nstack_preds = nstack.predict(test_features, y_hat_base_models_test)
                # NSTACK SCORE
                test_nstack_df = test_set.copy()
                test_nstack_df['y_hat'] = nstack_preds['stacking_prediction'].values
                predictions_df = test_nstack_df[['unique_id', 'y_hat', 'ds']]
                combination_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                                y_train_df=y_train_df,
                                                                y_test_df=test_nstack_df,
                                                                naive2_seasonality=seas_dict[seasonality]['seasonality'],
                                                                return_averages=False)
                total_combination_owa = record_comination_owas(combination_owa, total_combination_owa)
                combination_owa_median = np.median(combination_owa)
                combination_owa = np.mean(combination_owa)

            elif combination_type == 'Model Averaging':
                averaging_preds = y_hat_base_models_test.sum(axis=1)/y_hat_base_models_test.shape[1]
                # NSTACK SCORE
                test_averaging_df = test_set.copy()
                test_averaging_df['y_hat'] = averaging_preds.values
                predictions_df = test_averaging_df[['unique_id', 'y_hat', 'ds']]
                combination_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                                y_train_df=y_train_df,
                                                                y_test_df=test_averaging_df,
                                                                naive2_seasonality=seas_dict[seasonality]['seasonality'],
                                                                return_averages=False)
                total_combination_owa = record_comination_owas(combination_owa, total_combination_owa)
                combination_owa_median = np.median(combination_owa)
                combination_owa = np.mean(combination_owa)

            elif combination_type == 'nbeats':
                exo_count = train_feats.shape[2]
                if extra_model_train is not None:
                    exo_count += extra_model_train.shape[2]
                es = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=10,
                                                      restore_best_weights=True)
                nbeats = NBeatsNet(backcast_length=seas_dict[seasonality]['output_size'],
                                   forecast_length=seas_dict[seasonality]['output_size'],
                                   stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),#, NBeatsNet.SEASONALITY_BLOCK),
                                   nb_blocks_per_stack=8,  share_weights_in_stack=False, thetas_dim=(4,4),#(4, 4), #blocksper = 8, False share
                                   hidden_layer_units=64,#64,
                                   input_dim=1,#y_hat_base_models_train.shape[1],
                                   exo_dim=exo_count)
                nbeats.compile(loss=smape_loss, optimizer=Adam(learning_rate=0.0001))
                targets = train_set[['y']].values
                targets = np.reshape(targets, (train_set['unique_id'].nunique(),
                                               seas_dict[seasonality]['output_size']))
                if extra_model_train is not None:
                    train_feats = np.concatenate((train_feats, extra_model_train), axis=2)
                    test_feats = np.concatenate((test_feats, extra_model_test), axis= 2)
                nbeats.fit([y_hat_base_models_train, train_feats], targets, epochs=300, batch_size=20, callbacks=[es])
                nbeats_preds = nbeats.predict([y_hat_base_models_test, test_feats])
                nbeats_preds = nbeats_preds.flatten()
                # NBEATS SCORE
                test_nbeats_df = test_set.copy()
                test_nbeats_df['y_hat'] = nbeats_preds
                predictions_df = test_nbeats_df[['unique_id', 'y_hat', 'ds']]
                combination_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                                y_train_df=y_train_df,
                                                                y_test_df=test_nbeats_df,
                                                                naive2_seasonality=seas_dict[seasonality][
                                                                    'seasonality'],
                                                                return_averages=False)
                total_combination_owa = record_comination_owas(combination_owa, total_combination_owa)
                combination_owa_median = np.median(combination_owa)
                combination_owa = np.mean(combination_owa)


            # ESRNN SCORE
            test_esrnn_df = test_set.copy()
            test_esrnn_df = test_esrnn_df.rename(columns={'mdl_ESRNN': "y_hat"})
            predictions_df = test_esrnn_df[['unique_id', 'y_hat', 'ds']]
            esrnn_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                      y_train_df=y_train_df,
                                                      y_test_df=test_esrnn_df,
                                                      naive2_seasonality=seas_dict[seasonality]['seasonality'],
                                                      return_averages=False)

            combination_run_loss += combination_owa
            combination_run_loss_median += combination_owa_median
            esrnn_run_loss += np.average(esrnn_owa)

            print(15 * '=','RUN:',run_num+1, ' FOLD:',test_fold_num+1, 15 * '=')
            print('ESRNN OWA ', np.average(esrnn_owa))
            print(combination_type +' Average OWA', combination_owa)
            print(combination_type + ' Median OWA', combination_run_loss_median)

        esrnn_run_score = esrnn_run_loss / k_folds
        fforma_run_score = combination_run_loss / k_folds
        overall_combination_loss += fforma_run_score
        overall_combination_loss_median += (combination_run_loss_median / k_folds)
        overall_esrnn_loss += esrnn_run_score

        print(15 * '=', 'RUN:', run_num + 1, ' AVERAGES:', 15 * '=')
        print('ESRNN OWA: {} '.format(np.round(esrnn_run_score, 3)))
        print(combination_type + ' Average OWA: {} '.format(np.round(fforma_run_score, 3)))
        print(combination_type + ' Median OWA: {} '.format(np.round((combination_run_loss_median / k_folds), 3)))

    print(15 * '=',  'AVERAGES OVER ALL RUNS:', 15 * '=')
    print('ESRNN OWA: {} '.format(np.round(overall_esrnn_loss/n_runs, 3)))
    print(combination_type + ' Average OWA: {} '.format(np.round(overall_combination_loss / n_runs, 3)))
    print(combination_type + ' Median OWA: {} '.format(np.round(overall_combination_loss_median / n_runs, 3)))

    # SAVE combo OWAs to file
    with open('results/'+combination_type+'_'+seasonality[0]+'.npy', 'wb') as f:
        np.save(f, total_combination_owa)

if __name__ == '__main__':
    seasonality = 'Hourly'
    X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(seasonality, 'data', 'forecasts', load_existing_dataframes=True)
    run(df_info=X_test_df,
        df_train_data=y_train_df,
        df_pred_data=y_test_df,
        seasonality=seasonality,
        optimizing_runs=0,
        combination_type='Neural Averaging 2',
        n_runs=5,
        k_folds=10,
        hyper_search_run=False
        )
