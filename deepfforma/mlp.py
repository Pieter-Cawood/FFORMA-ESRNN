from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
from operator import itemgetter


def fforma_loss(y_OWA, y_wm):
    return tf.reduce_sum(y_wm*y_OWA, axis=-1)

def Conv_1D_Block(x, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, padding='same', kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

class SumOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w / tf.reduce_sum(w)
    
    @classmethod
    def init(cls, shape, dtype):
        w = tf.random.normal(shape, dtype=dtype)
        w = w / tf.reduce_sum(w)
        return w

class SumZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w - tf.reduce_mean(w)

    @classmethod
    def init(cls, shape, dtype):
        w = tf.random.normal(shape, dtype=dtype)
        w = w - tf.reduce_mean(w)
        return w

def VGG_11(length, num_channel, num_filters, dropout_rate, min_length, seasons):
    inputs = tf.keras.Input((length, num_channel))  # The input tensor
    
    #Moving average and lagging head
    xt = inputs
    xt = tf.keras.layers.Conv1D(num_filters, (seasons*2-1),
                                padding='valid', 
                                kernel_initializer=SumOne.init,
                                kernel_constraint=SumOne(),
                                use_bias=False)(xt)
    xt = tf.keras.layers.LayerNormalization(axis=1)(xt)
    xt = tf.keras.layers.ZeroPadding1D(padding=(seasons-1,seasons-1))(xt)
    xt = tf.keras.layers.SpatialDropout1D(dropout_rate)(xt)

    #Differencing head
    xs = inputs
    xs = tf.keras.layers.ZeroPadding1D(padding=(0,seasons-1))(xs)
    xs = tf.keras.layers.Conv1D(num_filters, (seasons),
                                padding='valid',
                                kernel_initializer=SumZero.init,
                                kernel_constraint=SumZero(),
                                use_bias=False)(xs)
    xs = tf.keras.layers.LayerNormalization(axis=1)(xs)
    # xs = tf.keras.layers.ZeroPadding1D(padding=(seasons,seasons))(xs)
    xs = tf.keras.layers.SpatialDropout1D(dropout_rate)(xs)

    x = tf.keras.layers.Concatenate(axis=2)([xt,xs])
    
    # Block 1
    x = Conv_1D_Block(x, num_filters * (2 ** 0), 3)
    if min_length >= 10:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    # xa = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Block 2
    x = Conv_1D_Block(x, num_filters * (2 ** 1), 3)
    if min_length >= 16:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    # xb = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Block 3
    x = Conv_1D_Block(x, num_filters * (2 ** 2), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 2), 3)
    if min_length >= 24:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    # xc = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Block 4
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    if min_length >= 32:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    # xd = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Block 5
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    
    x = tf.keras.layers.GlobalMaxPooling1D()(x) #Global Averaging replaces Flatten
    # x = tf.keras.layers.GlobalAveragePooling1D()(x) #Global Averaging replaces Flatten
    # xe= x    
    
    # x = tf.keras.layers.Concatenate(axis=1)([xe])

    # Create model.    
    return inputs, x

def is_test(x, y):
    return x % 10 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y

l2_normalise = tf.keras.layers.UnitNormalization()
# z_normalise = tf.keras.layers.LayerNormalization()

def preprocessing(inp, max_length, min_length):
    inp = l2_normalise(inp)
    inp = inp if max_length is None else inp[-max_length:]
    pad_size = min_length - tf.shape(inp)[0] if min_length > tf.shape(inp)[0] else 0
    paddings = [[0, pad_size]]
    inp = tf.pad(inp, paddings)
    return inp #rememebr you changed this after you ran the experiment, used to be inp = z_normalise(inp)

class DeepFFORMA():
    def __init__(self, mc, n_features, n_models):
        self.mc = mc
        self.n_features = n_features
        self.n_models = n_models
        self.seasons  = self.mc['model_parameters']['seasons']
        self.min_length  = self.mc['model_parameters']['min_length']
        vgg_filters = self.mc['model_parameters']['vgg_filters']
        dropout_rate = self.mc['model_parameters']['dropout_rate']
        lr = self.mc['train_parameters']['learn_rate']
        self.batch_size = self.mc['train_parameters']['batch_size']
        self.max_length  = self.mc['train_parameters']['max_length']

        layer_units = [(vgg_filters * (2 ** 3))*4,(vgg_filters * (2 ** 3))*4]
        # layer_units_iid = [(vgg_filters * (2 ** 3)),(vgg_filters * (2 ** 3))]
        
        #The Features Model
        # inputs_iid = tf.keras.Input((n_features, ))  # The input tensor        
        # outputs_iid = tf.keras.layers.Layer()(inputs_iid)
        # outputs_iid = tf.keras.layers.Dropout(dropout_rate)(inputs_iid)
        # for _i in range(len(layer_units_iid)):
        #     outputs_iid = tf.keras.layers.Dense(layer_units_iid[_i],activation='relu')(outputs_iid)
        #     outputs_iid = tf.keras.layers.BatchNormalization()(outputs_iid)
        # outputs_iid = tf.keras.layers.Dense(n_models, activation='tanh')(outputs_iid)

        #The Time Series Model
        # inputs_ts = tf.keras.Input((None, ))  # The input tensor
        # outputs_ts = tf.keras.layers.Layer()(inputs_ts) 
        inputs_ts, outputs_ts = VGG_11(None, 1, vgg_filters, dropout_rate, self.min_length, self.seasons)
        self.features = tf.keras.Model(inputs_ts, outputs_ts)
        for _i in range(len(layer_units)):
            outputs_ts = tf.keras.layers.Dense(layer_units[_i],activation='relu')(outputs_ts)
            if dropout_rate:
                outputs_ts = tf.keras.layers.Dropout(dropout_rate)(outputs_ts)
        outputs_ts = tf.keras.layers.Dense(n_features,activation='linear')(outputs_ts)
        outputs_ts = tf.keras.layers.BatchNormalization()(outputs_ts)

        #Join the two models together
        outputs = outputs_ts #tf.keras.layers.Concatenate(axis=1)([outputs_ts])
        outputs = tf.keras.layers.Dense(n_models, 
                                        use_bias=False,
                                        activation='softmax')(outputs)

        self.optimizer = Adam(lr=lr)
        self.model = tf.keras.Model(inputs=inputs_ts, outputs=outputs)
        self.model.compile(loss=fforma_loss, optimizer=self.optimizer)
        # self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
        #                    optimizer=self.optimizer)
        
        # self.output_types =(tf.float32, tf.float32)
        # self.output_shapes=((None,), (self.n_features,))
        self.output_types =tf.float32
        self.output_shapes=(None,) 

        self.fitted = False

    def fit(self, series, train_feats, errors):
        batch_size = self.batch_size
        train_errors = errors
        ts_pred_data = series
        train_errors_ID = pd.to_numeric(train_errors.index.str[1:], errors='coerce')

        # train_errors = pd.get_dummies( np.argmin(train_errors.values, axis=1) )
        # train_errors = (1/(train_errors+1e-9))/(1/(train_errors+1e-9)).sum(axis=1).to_frame().values
        # train_errors = train_errors/train_errors.sum(axis=1).to_frame().values

        gen_series =  [(preprocessing(ts, self.max_length, self.min_length), trg)
                            for ts, trg in 
                                zip(itemgetter(*train_errors_ID)(ts_pred_data),
                                # train_feats.loc[train_errors.index].values,
                                train_errors.loc[train_errors.index].values)]

        ds_series_validate = tf.data.Dataset.from_generator(
                lambda: gen_series,
                output_types =(self.output_types, tf.float32), 
                output_shapes=(self.output_shapes, (self.n_models,)))
        
        ds_series_train = tf.data.Dataset.from_generator(
                lambda: gen_series,
                output_types =(self.output_types, tf.float32),
                output_shapes=(self.output_shapes, (self.n_models,)))
        

        validate_dataset = ds_series_validate.enumerate() \
                            .filter(is_test) \
                            .map(recover, num_parallel_calls=tf.data.AUTOTUNE) \
                            .padded_batch(batch_size)

        train_dataset    = ds_series_train.enumerate() \
                            .filter(is_train) \
                            .map(recover, num_parallel_calls=tf.data.AUTOTUNE) \
                            .shuffle(self.batch_size*10) \
                            .padded_batch(batch_size)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=self.mc['train_parameters']['stop_grow_count'],
                                              restore_best_weights=True)

        epochs=self.mc['train_parameters']['epochs']

        self.model.fit(train_dataset,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[es],
                       validation_data=validate_dataset,
                       validation_freq=1,
                       use_multiprocessing=True,
                       workers=36                       
                       )

        self.fitted = True

    def predict(self, series, test_feats, y_hat_df):
        ts_pred_data = series
        assert self.fitted, 'Model not yet fitted'
        
        uids = y_hat_df.reset_index().unique_id.drop_duplicates()
        test_set_ID = pd.to_numeric(uids.str[1:], errors='coerce')
        #    test_feats.loc[uids].values)
        #    )

        gen_series = [(preprocessing(ts, self.max_length, self.min_length),) for ts in itemgetter(*test_set_ID)(ts_pred_data)]
        ds_series_test = tf.data.Dataset.from_generator(
                lambda: gen_series,
                output_types =(self.output_types,), 
                output_shapes=(self.output_shapes,))
        test_dataset = ds_series_test.padded_batch(self.batch_size)
        predicted_weights = self.model.predict(test_dataset)
        weights_all = dict((k,v) for k,v in zip(uids, predicted_weights))
        # print(weights_all)
        predicted_weights = np.concatenate([[weights_all[uid]] for uid in y_hat_df.reset_index().unique_id])
        print(predicted_weights)
        fforma_preds = predicted_weights * y_hat_df
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = 'navg_prediction'
        preds = pd.concat([y_hat_df, fforma_preds], axis=1)
        return preds





