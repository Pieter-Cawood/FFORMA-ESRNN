from scipy.stats import spearmanr
from utils.data import *
from utils.analysis import evaluate_prediction_owa

def compute_feature_correlation(seasonality, model_name, y_train_df, y_test_df, category=None):
    y_test_df = y_test_df.rename(columns={model_name:"y_hat"})
    if category is not None:
        y_test_df = y_test_df[y_test_df[category] == 1]
    predictions_df = y_test_df[['unique_id', 'y_hat', 'ds']]
    model_owa, _, _ = evaluate_prediction_owa(predictions_df=predictions_df,
                                              y_train_df=y_train_df,
                                              y_test_df=y_test_df,
                                              naive2_seasonality=seas_dict[seasonality]['seasonality'],
                                              return_averages=False)
    y_test_df = y_test_df.drop_duplicates('unique_id')
    for col in y_test_df.columns:
        if col.startswith('mf_'):
            corr = spearmanr(y_test_df[col].values, model_owa)
            print(col, ' corr : ',abs(corr[0]))

if __name__ == '__main__':
    seasonality = 'Daily'
    X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(seasonality, 'data', 'forecasts',
                                                             load_existing_dataframes=True)
    compute_feature_correlation(seasonality, 'mdl_esrnn', y_train_df, y_test_df)
