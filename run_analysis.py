from scipy.stats import spearmanr
from utils.data import *
from utils.analysis import evaluate_prediction_owa
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
from copy import deepcopy

if __name__ == '__main__':
    seasonality = 'Weekly'
    max_owa_clip = 3.0
  #  test = np.load('/tmp/123.npy')

    X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(seasonality, 'data', 'forecasts',
                                                             load_existing_dataframes=True)
    fig, ax = plt.subplots(1, 2)

    os.chdir('results')
    f = open(seasonality[0]+"_summary.txt", "w")
    base_errors = y_test_df.copy()
    base_errors = base_errors.drop_duplicates('unique_id').set_index('unique_id')
    base_errors = base_errors.filter(regex='^OWA_', axis=1)
    base_errors.columns = base_errors.columns.str.lstrip('OWA_')
    base_errors.columns = base_errors.columns.str.lstrip('mdl_')
    plot_base_errors = deepcopy(base_errors)
    plot_base_errors[plot_base_errors > max_owa_clip] = max_owa_clip

    plot_base_errors.rename(columns={'ESRNN': 'ES-RNN'},
                            inplace=True)

    sns.violinplot(data=plot_base_errors, ax=ax[0])
    ax[0].tick_params(axis='x', rotation=35)
   # base_ax.set_title(seasonality + ' Data - Base Model Forecasts')
    ax[0].set_ylabel('OWA')
  #  plt.savefig(seasonality[0] + '_basemodels_plot.pdf')
   # plt.show()


    for mdl, avg_owa in zip(base_errors.columns, base_errors.mean()):
        str_result = mdl + " average : " + str(avg_owa)
        print(str_result)
        f.write(str_result + '\n')

    ensemble_names = []
    ensemble_owas = None
    for file in glob.glob("*"+seasonality[0]+".npy"):
        model_name = file.rsplit('_'+seasonality[0]+'.npy')[0]
        ensemble_names.append(model_name)
        owas = np.load(file)
        if ensemble_owas is None:
            ensemble_owas = pd.DataFrame(owas,columns=[model_name])
        else:
            ensemble_owas[model_name] = owas
        str_result = model_name + " average : " + str(np.mean(owas))
        print(str_result)
        f.write(str_result + '\n')
      #  print(model_name, "max :", np.max(owas))

    ensemble_owas.rename(columns={'Model Averaging': 'AVG',
                                  'Neural Averaging': 'NN-AVG',
                                  'Neural Stacking': 'NN-STACK',
                                  },
                         inplace=True)
    plot_ensemble_errors = deepcopy(ensemble_owas)
    plot_ensemble_errors[plot_ensemble_errors > max_owa_clip] = max_owa_clip
    sns.violinplot(ax=ax[1], data=plot_ensemble_errors, palette={"FFORMA": "y", "FFORMS":"lime", "AVG":"white", "NN-AVG": "pink", "NN-STACK":"grey"})
    ax[1].tick_params(axis='x', rotation=35)
    #ensemble_ax.set_title(seasonality + ' Data - Ensemble Forecasts')
   # ax[1].set_ylabel('OWA')
    plt.subplots_adjust(bottom=0.15)
    fig.tight_layout()
    fig.savefig(seasonality[0] + '_violin_plots.pdf')
    #plt.show()
    fig.show()

    f.close()

