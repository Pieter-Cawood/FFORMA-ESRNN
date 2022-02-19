from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, ReLU, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import pandas as pd

class StackingMLP():
    def __init__(self,mc, n_features, style='Neural Stacking'):
        self.mc = mc#{
     #       'model_parameters': {
    #            'layer_units': [10, 10, 10]#, 10, 10]#, 50, 50, 50]
   #         },
   #         'train_parameters': {
   #             'learn_rate': 3e-4,
   #             'batch_size': 225,
   #             'epochs': 300,
   #             'stop_grow_count': 15
    #        }
   #     }
        layer_units = self.mc['model_parameters']['layer_units']
        self.model = Sequential()
        # for n, units in enumerate(layer_units):
        self.model.add(Input([n_features]))
        if style == 'Neural Stacking':
            for _i in range(len(layer_units)):
                self.model.add(Dense(layer_units[_i]))
        elif style == 'Neural Stacking 2':
            self.model.add(Dense(layer_units[0], kernel_regularizer=l1_l2(l1=0.2, l2=0.5)))
            for _i in range(1,len(layer_units)):
                self.model.add(Dense(layer_units[_i]))
            # self.model.add(BatchNormalization())
        self.model.add(Dense(1))
        #   lr_schedule = ExponentialDecay(
        #        initial_learning_rate=self.mc['train_parameters']['learn_rate'],
        #      decay_steps=self.mc['train_parameters']['decay_steps'],
        #      decay_rate=self.mc['train_parameters']['gamma'])
        self.optimizer = Adam(lr=self.mc['train_parameters']['learn_rate'])
        self.model.compile(loss='mae', optimizer=self.optimizer)

        self.fitted = False

    def fit(self, features, targets):

        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.mc['train_parameters']['stop_grow_count'],
                                              restore_best_weights=True)
        self.model.fit(features,
                       targets,
                       batch_size=self.mc['train_parameters']['batch_size'],
                       epochs=self.mc['train_parameters']['epochs'],
                       verbose=1,
                       callbacks=[es])
        self.fitted = True

    def predict(self, features, y_hat_df):
        assert self.fitted, 'Model not yet fitted'
        y_hat = self.model.predict(features)
        predictions_df = y_hat_df.copy()
        predictions_df['stacking_prediction'] = y_hat
     #   fforma_preds = weights * y_hat_df
     #   fforma_preds = fforma_preds.sum(axis=1)
      #  fforma_preds.name = 'fforma_prediction'
      #  preds = pd.concat([y_hat_df, fforma_preds], axis=1)
        return predictions_df