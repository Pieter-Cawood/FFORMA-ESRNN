import pandas as pd
import numpy as np
import glob, os
from collections import defaultdict
from copy import deepcopy
from tsfeatures import tsfeatures
from natsort import natsort_keygen, natsorted
from .analysis import evaluate_prediction_owa
from sklearn.preprocessing import LabelEncoder

PIETER_TEST = False
seas_dict = {'Hourly': {'seasonality': 24, 'input_size': 24,
                       'output_size': 48, 'freq': 'H',
                        'tail_inputs': 0},
             'Daily': {'seasonality': 7, 'input_size': 7,
                       'output_size': 14, 'freq': 'D',
                       'tail_inputs': 0},
             'Weekly': {'seasonality': 52, 'input_size': 52,
                        'output_size': 13, 'freq': 'W',
                        'tail_inputs': 0},
             'Monthly': {'seasonality': 12, 'input_size': 12,
                         'output_size':18, 'freq': 'M',
                         'tail_inputs': 0},
             'Quarterly': {'seasonality': 4, 'input_size': 4,
                           'output_size': 8, 'freq': 'Q',
                           'tail_inputs': 0},
             'Yearly': {'seasonality': 1, 'input_size': 4,
                        'output_size': 6, 'freq': 'D',
                        'tail_inputs': 0}}

FREQ_TO_INT = {'H': 24, 'D': 1,
               'M': 12, 'Q': 4,
               'W':1, 'Y': 1}

def extract_meta_features(seas_dict, dataset_name, y_train_df, y_test_df):
    tail_inputs = seas_dict[dataset_name]['tail_inputs']
    if tail_inputs > 0:
        y_train_df = y_train_df.groupby('unique_id').tail(tail_inputs)
    meta_features = tsfeatures(y_train_df, FREQ_TO_INT[seas_dict[dataset_name]['freq']])
    # Sort unique_ids naturally/alphanumerically so we can concatenate later
    meta_features = meta_features.sort_values(
        by="unique_id",
        key=natsort_keygen()
    )
    #Drop all nan columns, make other nans = zeros
    meta_features = meta_features.dropna(axis=1, how='all').fillna(0).add_prefix('mf_')
    meta_features = meta_features.rename(columns={"mf_unique_id": "unique_id"})
    #Repeat features for every point of forecast horizon
    # (The same reference series' statistics were present for each point.)
    meta_features = meta_features.loc[meta_features.index.repeat(seas_dict[dataset_name]['output_size'])]
    y_test_df = pd.concat([y_test_df, meta_features.drop('unique_id', axis=1).reset_index(drop=True)], axis=1)
    return y_train_df, y_test_df

def compute_model_errors(seas_dict, base_model_names, dataset_name, y_train_df, y_test_df):
    print("Calculating model errors")
    errors_train_df = pd.DataFrame({'unique_id': natsorted(y_test_df.unique_id.unique())}).set_index('unique_id')
    for mdl in base_model_names:
        if mdl != 'mdl_naive2':
            train_set_mdl = y_test_df.rename(columns={mdl: "y_hat"})
            predictions_df = train_set_mdl[['unique_id', 'y_hat', 'ds']]
            model_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                                      y_train_df=y_train_df,
                                                      y_test_df=train_set_mdl,
                                                      naive2_seasonality=seas_dict[dataset_name]['seasonality'],
                                                      return_averages=False)
            errors_train_df['OWA_'+ mdl] = model_owa
    errors_train_df = errors_train_df.loc[errors_train_df.index.repeat(seas_dict[dataset_name]['output_size'])]
    y_test_df = pd.concat([y_test_df, errors_train_df.reset_index(drop=True)], axis=1)
    return y_test_df

def m4_parser(dataset_name, data_directory, forecast_directory, load_existing_dataframes=True):
  """
  Transform M4 data into a panel.

  Parameters
  ----------
  dataset_name: str
    Frequency of the data. Example: 'Yearly'.
  directory: str
    Custom directory where data will be saved.
  num_obs: int
    Number of time series to return.
  """
  # Load previously computed dataframes
  if load_existing_dataframes and os.path.isfile(data_directory + "/Preprocessed/" + dataset_name + '_X_train_df.csv'):
      X_train_df = pd.read_csv(data_directory + "/Preprocessed/" + dataset_name + '_X_train_df.csv')
      y_train_df = pd.read_csv(data_directory + "/Preprocessed/" + dataset_name + '_y_train_df.csv')
      X_test_df = pd.read_csv(data_directory + "/Preprocessed/" + dataset_name + '_X_test_df.csv')
      y_test_df = pd.read_csv(data_directory + "/Preprocessed/" + dataset_name + '_y_test_df.csv')
      return X_train_df, y_train_df, X_test_df, y_test_df

  print("Processing data")
  train_directory = data_directory + "/Train/"
  test_directory = data_directory + "/Test/"
  freq = seas_dict[dataset_name]['freq']

  m4_info = pd.read_csv(data_directory+'/M4-info.csv', usecols=['M4id','category'])
  m4_info = m4_info[m4_info['M4id'].str.startswith(dataset_name[0])].reset_index(drop=True)

  # Train data
  train_path='{}{}-train.csv'.format(train_directory, dataset_name)

  train_df = pd.read_csv(train_path)
  train_df = train_df.rename(columns={'V1':'unique_id'})

  train_df = pd.wide_to_long(train_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
  train_df = train_df.rename(columns={'V':'y'})
  train_df = train_df.dropna()
  train_df['split'] = 'train'
  train_df['ds'] = train_df['ds']-1
  # Get len of series per unique_id
  len_series = train_df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
  len_series.columns = ['unique_id', 'len_serie']

  # Test data
  test_path='{}{}-test.csv'.format(test_directory, dataset_name)

  test_df = pd.read_csv(test_path)
  test_df = test_df.rename(columns={'V1':'unique_id'})

  test_df = pd.wide_to_long(test_df, stubnames=["V"], i="unique_id", j="ds").reset_index()
  test_df = test_df.rename(columns={'V':'y'})
  test_df = test_df.dropna()
  test_df['split'] = 'test'
  test_df = test_df.merge(len_series, on='unique_id')
  test_df['ds'] = test_df['ds'] + test_df['len_serie'] - 1
  test_df = test_df[['unique_id','ds','y','split']]

  df = pd.concat((train_df,test_df)).reset_index(drop=True)

  df = df.sort_values(by=['unique_id', 'ds'], key=natsort_keygen()).reset_index(drop=True)

  # Create column with dates with freq of dataset
  len_series = df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
  len_series = len_series.sort_values(by=['unique_id'], key=natsort_keygen())
  dates = []
  for i in range(len(len_series)):
      len_serie = len_series.iloc[i,1]
      ranges = pd.date_range(start='1970/01/01', periods=len_serie, freq=freq)
      dates += list(ranges)

  df.loc[:,'ds'] = dates

  df = df.merge(m4_info, left_on=['unique_id'], right_on=['M4id'])
  df.drop(columns=['M4id'], inplace=True)
 # df = df.rename(columns={'category': 'x'})

  X_train_df = df[df['split']=='train'].filter(items=['unique_id', 'ds', 'category'])
  y_train_df = df[df['split']=='train'].filter(items=['unique_id', 'ds', 'y'])
  X_test_df = df[df['split']=='test'].filter(items=['unique_id', 'ds', 'category'])
  y_test_df = df[df['split']=='test'].filter(items=['unique_id', 'ds', 'y'])

  X_train_df = X_train_df.reset_index(drop=True)
  y_train_df = y_train_df.reset_index(drop=True)
  X_test_df = X_test_df.reset_index(drop=True)
  y_test_df = y_test_df.reset_index(drop=True)

  # Add forecast models' predictions
  os.chdir(forecast_directory)
  base_model_names = []
  for file in glob.glob("*.csv"):
      model_name = 'mdl_' + file.rsplit('.csv')[0]
      base_model_names.append(model_name)
      model_df = pd.read_csv(file)
      model_df = model_df.rename(columns={'id': 'unique_id'})
      model_df = model_df[model_df['unique_id'].str.startswith(dataset_name[0])]
      #TODO this a bit dirty, need to cut after horizon instead
      model_df = model_df.dropna(axis=1)
      model_forecasts = {'unique_id':[], model_name:[]}
      for u_id in natsorted(model_df.unique_id.unique()):
          values = model_df[model_df['unique_id'] == u_id].values[0][1:]
          model_forecasts['unique_id'].extend([u_id]*len(values))
          model_forecasts[model_name].extend(values)
      model_df = pd.DataFrame(model_forecasts)
      y_test_df[model_name] = model_df[model_name].values

  # One hot encodings of domain to df
  one_hot_encodings = pd.get_dummies(X_test_df.category, prefix='category')
  y_test_df = pd.concat([y_test_df, one_hot_encodings], axis=1)
  #LE = LabelEncoder()
  #y_test_df['category'] = LE.fit_transform(X_test_df['category'])

  y_train_df, y_test_df = extract_meta_features(seas_dict, dataset_name, y_train_df, y_test_df)

  y_test_df = compute_model_errors(seas_dict, base_model_names, dataset_name, y_train_df, y_test_df)

  os.chdir(os.path.split(os.getcwd())[0] + '/' + data_directory + "/Preprocessed/")
  X_train_df.to_csv(dataset_name + '_X_train_df.csv')
  y_train_df.to_csv(dataset_name + '_y_train_df.csv')
  X_test_df.to_csv(dataset_name + '_X_test_df.csv')
  y_test_df.to_csv(dataset_name + '_y_test_df.csv')

  print("Finished processing data")
  return X_train_df, y_train_df, X_test_df, y_test_df

def get_number_of_categories(X_test_df):
    return len(X_test_df['category'].unique())

def make_kfolds(df_info, df_data, k_folds, seed):
    """
    Splits data into n number of categorically-balanced folds

    :param data:
    :param k_folds:
    :return:
    """
    #data['idx'] = data.index
    kfold_ids = defaultdict(list)
    for category in natsorted(df_info.category.unique()):
        cat_idxs = natsorted(df_info[df_info['category'] == category].unique_id.unique())
        np.random.seed(seed)
        np.random.shuffle(cat_idxs)
        split_idxs = np.array_split(cat_idxs, k_folds)
        for i in range(k_folds):
            kfold_ids[i].extend(split_idxs[i])
    kfoldings = []
    for fold_num in range(k_folds):
      kfoldings.append(df_data[df_data['unique_id'].isin(kfold_ids[fold_num])])
    return kfoldings

def train_test_split(kfoldings, test_fold_num=0):
    """
    Splits foldings into a train and test set according to iteration

    :param iteration:
    :return:
    """
    assert test_fold_num < len(kfoldings), "test_fold_num exceeds the size of kfoldings"

    test_set = kfoldings[test_fold_num]
    train_set = None
    for fold_num in range(len(kfoldings)):
        if fold_num != test_fold_num:
            if train_set is None:
                train_set = kfoldings[fold_num]
            else:
                train_set = train_set.append(kfoldings[fold_num])
    return train_set, test_set

