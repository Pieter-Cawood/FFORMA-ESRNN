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

def TemporalHeads(length, num_channel, num_filters, dropout_rate, seasons):
    inputs = tf.keras.Input((length, num_channel))  # The input tensor
    xi = inputs
    
    #Moving average and lagging head
    xt = inputs
    xt = tf.keras.layers.Conv1D(num_filters, (seasons*2-1),
                                padding='valid', 
                                kernel_initializer=SumOne.init,
                                kernel_constraint=SumOne(),
                                use_bias=False)(xt)
    xt = tf.keras.layers.LayerNormalization(axis=1, 
                                            epsilon=1e-8, 
                                            center=False, 
                                            scale=False)(xt)
    xt = tf.keras.layers.ZeroPadding1D(padding=(seasons-1,seasons-1))(xt)
    xt = tf.keras.layers.SpatialDropout1D(dropout_rate)(xt)

    #Differencing head
    xr = inputs    
    xr = tf.keras.layers.Conv1D(num_filters, (seasons),
                                padding='valid',
                                kernel_initializer=SumZero.init,
                                kernel_constraint=SumZero(),
                                use_bias=False)(xr)
    xr = tf.keras.layers.LayerNormalization(axis=1,
                                            epsilon=1e-8, 
                                            center=False, 
                                            scale=False)(xr)
    xr = tf.keras.layers.ZeroPadding1D(padding=(0,seasons-1))(xr)
    xr = tf.keras.layers.SpatialDropout1D(dropout_rate)(xr)

    x = tf.keras.layers.Concatenate(axis=2)([xt,xr,xi])

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

def resnet(x, blocks_per_layer, num_filters, n_features, min_length):
    x = tf.keras.layers.ZeroPadding1D(padding=3, name='conv1_pad')(x)
    adstride = 2 if min_length >= 1568 else 1
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=7, strides=adstride, use_bias=False, kernel_initializer="he_normal", name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.ZeroPadding1D(padding=1, name='maxpool_pad')(x)
    if min_length >= 784:
        x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, num_filters * (2 ** 0), blocks_per_layer[0], name='layer1')
    adstride = 2 if min_length >= 392 else 1
    x = make_layer(x, num_filters * (2 ** 1), blocks_per_layer[1], stride=adstride, name='layer2')
    adstride = 2 if min_length >= 196 else 1
    x = make_layer(x, num_filters * (2 ** 2), blocks_per_layer[2], stride=adstride, name='layer3')
    adstride = 2 if min_length >= 98 else 1
    x = make_layer(x, num_filters * (2 ** 3), blocks_per_layer[3], stride=adstride, name='layer4')

    x = tf.keras.layers.GlobalMaxPooling1D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=n_features, name='fc')(x)

    return x

def resnet10(x, num_filters, n_features, min_length, **kwargs):
    return resnet(x, [1, 1, 1, 1], num_filters, n_features, min_length, **kwargs)

def resnet18(x, num_filters, n_features, min_length, **kwargs):
    return resnet(x, [2, 2, 2, 2], num_filters, n_features, min_length, **kwargs)

def Conv_1D_Block(x, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, padding='same', kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def VGG_11(x, num_filters, n_features, min_length, dropout_rate):
    
    # Block 1
    x = Conv_1D_Block(x, num_filters * (2 ** 0), 3)
    if min_length >= 1568:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 2
    x = Conv_1D_Block(x, num_filters * (2 ** 1), 3)
    if min_length >= 784:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 3
    x = Conv_1D_Block(x, num_filters * (2 ** 2), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 2), 3)
    if min_length >= 392:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 4
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    if min_length >= 196:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

    # Block 5
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    x = Conv_1D_Block(x, num_filters * (2 ** 3), 3)
    if min_length >= 98:
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

def is_test(x, y):
    return x % 10 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y

z_normalise = tf.keras.layers.LayerNormalization(axis=0,
                                                 epsilon=1e-8, 
                                                 center=False, 
                                                 scale=False)

def preprocessing(inp, max_length, min_length):
    inp = inp if max_length is None else inp[-max_length:]
    inp = z_normalise(inp)
    pad_size = min_length - tf.shape(inp)[0] if min_length > tf.shape(inp)[0] else 0
    paddings = [[0, pad_size]]
    inp = tf.pad(inp, paddings)
    return inp

class DeepFFORMA():
    def __init__(self, mc, n_features, n_models):
        self.mc = mc
        self.n_features = n_features
        self.n_models = n_models
        self.seasons  = self.mc['model_parameters']['seasons']
        self.min_length  = self.mc['model_parameters']['min_length']
        vgg_filters = self.mc['model_parameters']['vgg_filters']
        res_filters = self.mc['model_parameters']['res_filters']
        dropout_rate = self.mc['model_parameters']['dropout_rate']
        lr = self.mc['train_parameters']['learn_rate']
        self.batch_size = self.mc['train_parameters']['batch_size']
        self.max_length  = self.mc['train_parameters']['max_length']

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
        if vgg_filters is not None:
            inputs_ts, outputs_ts = TemporalHeads(None, 1, vgg_filters, dropout_rate, self.seasons)
            outputs_ts = VGG_11(outputs_ts, n_features, vgg_filters, self.min_length, dropout_rate)            
        elif res_filters is not None:
            inputs_ts, outputs_ts = TemporalHeads(None, 1, res_filters, dropout_rate, self.seasons)
            outputs_ts = resnet10(outputs_ts, n_features, res_filters, self.min_length)
        else:
            raise NotImplemented()
        
        self.features = tf.keras.Model(inputs_ts, outputs_ts)

        #Join the two models together
        outputs = outputs_ts #tf.keras.layers.Concatenate(axis=1)([outputs_ts])
        outputs_ts = tf.keras.layers.BatchNormalization()(outputs)
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

        gen_series = [(preprocessing(ts, self.max_length, self.min_length), trg)
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
                       workers=16                       
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





