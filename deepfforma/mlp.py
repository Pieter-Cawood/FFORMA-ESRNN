from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
from operator import itemgetter

def fforma_loss(y_OWA, y_wm):
    return tf.reduce_sum(y_wm*y_OWA, axis=-1)

class SumOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w / tf.reduce_sum(w)
    
    @classmethod
    def init(cls, shape, dtype):
        w = tf.random.normal(shape, dtype=dtype)
        return w / tf.reduce_sum(w)

class SumZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w - tf.reduce_mean(w)

    @classmethod
    def init(cls, shape, dtype):
        w = tf.random.normal(shape, dtype=dtype)
        return w - tf.reduce_mean(w)

class SoftMax(tf.keras.constraints.Constraint):
    def __call__(self, w):
        exp = tf.exp(w)
        return exp / tf.reduce_sum(exp)

    @classmethod
    def init(cls, shape, dtype):
        w = tf.random.normal(shape, dtype=dtype)
        exp = tf.exp(w)
        return exp / tf.reduce_sum(exp)

def TemporalHeads(inputs, num_filters, dropout_rate, seasons):
    # inputs = tf.keras.layers.ZeroPadding1D(padding=(0,seasons))(inputs)

    # xi = inputs            
    # xi = tf.keras.layers.Conv1D(num_filters, (seasons+1),
    #                             padding='valid',
    #                             kernel_initializer=SumOne.init,
    #                             kernel_constraint=SumOne(),
    #                             use_bias=False)(xi)
    # xi = tf.keras.layers.LayerNormalization(axis=1, 
    #                                         epsilon=1e-8, 
    #                                         center=False, 
    #                                         scale=False)(xi)
    # xi = tf.keras.layers.SpatialDropout1D(dropout_rate)(xi)
    
    #Moving average head
    xt = inputs
    xt = tf.keras.layers.Conv1D(num_filters, (seasons+1),
                                padding='valid',
                                kernel_initializer=SoftMax.init,
                                kernel_constraint=SoftMax(),
                                use_bias=False)(xt)
    xt = tf.keras.layers.LayerNormalization(axis=1, 
                                            epsilon=1e-8, 
                                            center=False, 
                                            scale=False)(xt)
    xt = tf.keras.layers.SpatialDropout1D(dropout_rate)(xt)

    #Differencing head
    xr = inputs    
    xr = tf.keras.layers.Conv1D(num_filters, (seasons+1),
                                padding='valid',
                                kernel_initializer=SumZero.init,
                                kernel_constraint=SumZero(),
                                use_bias=False)(xr)
    xr = tf.keras.layers.LayerNormalization(axis=1,
                                            epsilon=1e-8, 
                                            center=False, 
                                            scale=False)(xr)
    xr = tf.keras.layers.SpatialDropout1D(dropout_rate)(xr)

    x = tf.keras.layers.Concatenate(axis=2)([xt,xr])

    return inputs, x

# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py

def conv1x3(x, out_planes, stride=1, name=None):
    x = tf.keras.layers.ZeroPadding1D(padding=1, name=f'{name}_pad')(x)
    x = tf.keras.layers.Conv1D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer="he_normal", name=name)(x)
    return tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{name}.bn')(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv1x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)

    out = conv1x3(out, planes, name=f'{name}.conv2')    

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = tf.keras.layers.Add(name=f'{name}.add')([identity, out])
    out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[2]
    if stride != 1 or inplanes != planes:
        downsample = [
            tf.keras.layers.Conv1D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer="he_normal", name=f'{name}.0.downsample.0'),
            tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_filters, n_features, halvings):
    x = tf.keras.layers.ZeroPadding1D(padding=3, name='conv1_pad')(x)
    adstride = 2 if halvings >= 1 else 1
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=7, strides=adstride, use_bias=False, kernel_initializer="he_normal", name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.ZeroPadding1D(padding=1, name='maxpool_pad')(x)
    if halvings >= 2:
        x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, num_filters * (2 ** 0), blocks_per_layer[0], name='layer1')
    adstride = 2 if halvings >= 3 else 1
    x = make_layer(x, num_filters * (2 ** 1), blocks_per_layer[1], stride=adstride, name='layer2')
    adstride = 2 if halvings >= 4 else 1
    x = make_layer(x, num_filters * (2 ** 2), blocks_per_layer[2], stride=adstride, name='layer3')
    adstride = 2 if halvings >= 5 else 1
    x = make_layer(x, num_filters * (2 ** 3), blocks_per_layer[3], stride=adstride, name='layer4')

    x = tf.keras.layers.GlobalMaxPooling1D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=n_features, name='fc')(x)

    return x

def resnet10(x, num_filters, n_features, halvings, **kwargs):
    return resnet(x, [1, 1, 1, 1], num_filters, n_features, halvings, **kwargs)

def resnet18(x, num_filters, n_features, halvings, **kwargs):
    return resnet(x, [2, 2, 2, 2], num_filters, n_features, halvings, **kwargs)

def Conv_1D_Block(x, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, padding='same', kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def VGG_11(x, num_filters, n_features, halvings, dropout_rate):
    
    # Block 1
    x = Conv_1D_Block(x, num_filters * (2 ** 0), 3)
    if halvings >= 1:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 2
    x = Conv_1D_Block(x, num_filters * (2 ** 1), 3)
    if halvings >= 2:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 3
    x = Conv_1D_Block(x, num_filters * (2 ** 2), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 2), 3)
    if halvings >= 3:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 4
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    if halvings >= 4:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 5
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    if halvings >= 5:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    
    x = tf.keras.layers.GlobalMaxPooling1D()(x) #Global Averaging replaces Flatten

    # Create model.    
    layer_units = [(num_filters * (2 ** 3))*4, (num_filters * (2 ** 3))*4]
    for _i in range(len(layer_units)):
        x = tf.keras.layers.Dense(layer_units[_i],activation='relu')(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(n_features,activation='linear')(x)
    
    return x

z_normalise = tf.keras.layers.LayerNormalization(axis=0,
                                                 epsilon=1e-8, 
                                                 center=False, 
                                                 scale=False)

def preprocessing(max_length, min_length, augment=False):
    def _preprocessing(inp, y):        
        if augment:
            aug = tf.random.uniform(shape=[], minval=0, maxval=augment, dtype=tf.int32)
            if aug > 0:
                inp = inp[:-aug]
        if max_length is not None:
            inp = inp[-max_length:]        
        inp = z_normalise(inp)
        if min_length > tf.shape(inp)[0]:
            pad_size = min_length - tf.shape(inp)[0]        
            paddings = [[0, pad_size]]
            inp = tf.pad(inp, paddings)
        return inp, y
    return _preprocessing

class DeepFFORMA():
    def __init__(self, mc, n_features, n_models):
        self.mc = mc                
        self.n_models = n_models

        self.min_length  = self.mc['model_parameters']['min_length']
        self.batch_size = self.mc['train_parameters']['batch_size']
        self.max_length  = self.mc['train_parameters']['max_length']        
        self.augment = self.mc['train_parameters']['augment']

        seasons  = self.mc['model_parameters']['seasons']
        halvings  = self.mc['model_parameters']['halvings']
        vgg_filters = self.mc['model_parameters']['vgg_filters']
        res_filters = self.mc['model_parameters']['res_filters']
        dropout_rate = self.mc['model_parameters']['dropout_rate']
        lr = self.mc['train_parameters']['learn_rate']
        
        inputs_ts = tf.keras.Input((None, 1))  # The input tensor

        if vgg_filters is not None:
            if seasons == 0:
                outputs_ts = inputs_ts
            else:
                inputs_ts, outputs_ts = TemporalHeads(inputs_ts, vgg_filters, dropout_rate, seasons)
            outputs_ts = VGG_11(outputs_ts, n_features, vgg_filters, halvings, dropout_rate)            
        elif res_filters is not None:
            if seasons == 0:
                outputs_ts = inputs_ts
            else:
                inputs_ts, outputs_ts = TemporalHeads(inputs_ts, res_filters, dropout_rate, seasons)
            outputs_ts = resnet10(outputs_ts, n_features, res_filters, halvings)
        else:
            raise NotImplemented()
        
        self.features = tf.keras.Model(inputs_ts, outputs_ts)

        outputs_ts = tf.keras.layers.BatchNormalization()(outputs_ts)
        outputs_ts = tf.keras.layers.Dense(n_models, 
                                            use_bias=False,
                                            activation='softmax')(outputs_ts)

        self.optimizer = Adam(lr=lr)
        self.model = tf.keras.Model(inputs=inputs_ts, outputs=outputs_ts)
        self.model.compile(loss=fforma_loss, optimizer=self.optimizer)
        # self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
        #                    optimizer=self.optimizer)
        
        self.output_types =tf.float32
        self.output_shapes=(None,) 

        self.fitted = False

    def fit(self, series, train_feats, errors):        
        train_errors = errors
        ts_pred_data = series
        train_errors_ID = pd.to_numeric(train_errors.index.str[1:], errors='coerce')

        # train_errors = pd.get_dummies( np.argmin(train_errors.values, axis=1) )
        # train_errors = (1/(train_errors+1e-9))/(1/(train_errors+1e-9)).sum(axis=1).to_frame().values
        # train_errors = train_errors/train_errors.sum(axis=1).to_frame().values

        gen_series_train = [(ts, trg)
                                for i, (ts, trg) in
                                    enumerate(
                                        zip(itemgetter(*train_errors_ID)(ts_pred_data),
                                        train_errors.loc[train_errors.index].values))
                                if i % 10 != 0]
        gen_series_valid = [(ts, trg)
                                for i, (ts, trg) in
                                    enumerate(
                                        zip(itemgetter(*train_errors_ID)(ts_pred_data),
                                        train_errors.loc[train_errors.index].values))
                                if i % 10 == 0]
        
        ds_series_train = tf.data.Dataset.from_generator(
                                lambda: gen_series_train,
                                output_types =(self.output_types, tf.float32),
                                output_shapes=(self.output_shapes, (self.n_models,)))
        
        ds_series_valid = tf.data.Dataset.from_generator(
                                lambda: gen_series_valid,
                                output_types =(self.output_types, tf.float32), 
                                output_shapes=(self.output_shapes, (self.n_models,)))

        preproc_train = preprocessing(self.max_length, self.min_length, self.augment)
        preproc_valid = preprocessing(self.max_length, self.min_length, False)

        train_dataset = ds_series_train.map(preproc_train, num_parallel_calls=tf.data.AUTOTUNE) \
                            .shuffle(self.batch_size*10) \
                            .padded_batch(self.batch_size) \
                            .prefetch(tf.data.AUTOTUNE)
        
        valid_dataset = ds_series_valid.map(preproc_valid, num_parallel_calls=tf.data.AUTOTUNE) \
                            .padded_batch(self.batch_size) \
                            .prefetch(tf.data.AUTOTUNE)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=self.mc['train_parameters']['stop_grow_count'],
                                              restore_best_weights=True)

        epochs=self.mc['train_parameters']['epochs']

        self.model.fit(train_dataset,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[es],
                       validation_data=valid_dataset,
                       validation_freq=1,
                       use_multiprocessing=True,
                       workers=32                       
                       )

        self.fitted = True

    def predict(self, series, test_feats, y_hat_df):
        ts_pred_data = series
        assert self.fitted, 'Model not yet fitted'
        
        uids = y_hat_df.reset_index().unique_id.drop_duplicates()
        test_set_ID = pd.to_numeric(uids.str[1:], errors='coerce')

        preproc = lambda inp: preprocessing(self.max_length, self.min_length, False)(inp, None)[0]

        gen_series = [(ts,) for ts in itemgetter(*test_set_ID)(ts_pred_data)]
        ds_series_test = tf.data.Dataset.from_generator(
                lambda: gen_series,
                output_types =(self.output_types,), 
                output_shapes=(self.output_shapes,))
        test_dataset = ds_series_test.map(preproc, num_parallel_calls=tf.data.AUTOTUNE) \
                                     .padded_batch(self.batch_size) \
                                     .prefetch(tf.data.AUTOTUNE)
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


