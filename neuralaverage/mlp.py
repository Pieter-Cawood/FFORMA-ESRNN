from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, ReLU, Dropout, Lambda, LayerNormalization, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import pandas as pd
import seaborn as sns

from utils.analysis import evaluate_prediction_owa

from scipy.special import softmax
''
#preds = np.reshape(predt,
#                   self.contribution_to_error[y, :].shape,
#                   order='F')
# lightgbm uses margins!
#
#
#fforma_loss = weighted_avg_loss_func.mean()
def fforma_loss(y_OWA, y_wm):
    return tf.reduce_sum(y_wm*y_OWA, axis=-1)

def sumoneact(x):
    # add small amount to ensure not all zero
    activated_x = K.relu(x) + 1e-3
    sumone_activated_x = activated_x/K.sum(activated_x, axis=1, keepdims=True)
    return sumone_activated_x


class ModelAveragingMLP():
    def __init__(self, mc, n_features, n_models, style='Neural Averaging'):
        self.mc = mc
        layer_units = self.mc['model_parameters']['layer_units']
        self.model = Sequential()
        #for n, units in enumerate(layer_units):
        self.model.add(Input([n_features]))        
        if style == 'Neural Averaging':
            for _i in range(len(layer_units)):
                self.model.add(Dense(layer_units[_i],activation='relu'))
                #self.model.add(BatchNormalization())
            self.model.add(Dense(n_models, activation='softmax'))
            self.optimizer = Adam(lr=self.mc['train_parameters']['learn_rate'])
            self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        elif style == 'Neural Averaging 2':
            for _i in range(len(layer_units)):
                self.model.add(Dense(layer_units[_i],activation='tanh'))
                self.model.add(BatchNormalization())
            # self.model.add(Dense(n_models,
            #                      activation=sumoneact,
            #                      use_bias=False,
            #                      kernel_regularizer=l1_l2(l1=1.0,l2=0.0)))
            self.model.add(Dense(n_models, activation='softmax', use_bias=False))
            self.optimizer = Adam(lr=self.mc['train_parameters']['learn_rate'])
            self.model.compile(loss=fforma_loss, optimizer=self.optimizer)

        self.fitted = False

    def fit(self, features, errors):
        contribution_to_error = errors.values
        best_models = contribution_to_error.argmin(axis=1)
        best_models_encoded = np.zeros((best_models.size, best_models.max() + 1))
        best_models_encoded[np.arange(best_models.size), best_models] = 1

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=self.mc['train_parameters']['stop_grow_count'],
                                              restore_best_weights=True)
        self.model.fit(features,
               #        best_models_encoded,
                       errors.values,
                       batch_size=self.mc['train_parameters']['batch_size'],
                       epochs=self.mc['train_parameters']['epochs'],
                       verbose=1,
                       callbacks=[es],
                       validation_split=0.15
                       )
        self.fitted = True

    def weights(self, features):
        assert self.fitted, 'Model not yet fitted'
        predicted_weights = self.model.predict(features)
        return predicted_weights

    def predict(self, features, y_hat_df):
        assert self.fitted, 'Model not yet fitted'
        predicted_weights = self.model.predict(features)
        weights = pd.DataFrame(predicted_weights,
                               index=features.index,
                               columns=y_hat_df.columns)       
        fforma_preds = weights * y_hat_df
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = 'navg_prediction'
        preds = pd.concat([y_hat_df, fforma_preds], axis=1)
        return preds