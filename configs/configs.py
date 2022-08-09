DEEPFFORMA_CONFIGS = {
    'Hourly': dict(
        model_parameters=dict(            
            min_length=32,
            vgg_filters=None,
            res_filters=16,
            dropout_rate=0.1,
            seasons=24
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=500,
            max_length=960,
            stop_grow_count=80
        )),
    
    'Daily': dict(
        model_parameters=dict(            
            min_length=32,
            vgg_filters=None,
            res_filters=24,
            dropout_rate=0.1,
            seasons=7
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=250,
            max_length=4197, #95
            stop_grow_count=40
        )),

    'Weekly': dict(
        model_parameters=dict(
            min_length=32,
            vgg_filters=None,
            res_filters=16,
            dropout_rate=0.1,
            seasons=52
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=500,
            max_length=4197, #95
            stop_grow_count=80
        )),
    
    'Monthly': dict(
        model_parameters=dict(
            min_length=32,
            vgg_filters=32,
            res_filters=None,
            dropout_rate=0.1,
            seasons=12
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=150,
            max_length=450, #95
            stop_grow_count=20
        )),

    'Quarterly': dict(
        model_parameters=dict(
            min_length=224,
            vgg_filters=64,
            res_filters=None,
            dropout_rate=0.1,
            seasons=4
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=150,
            max_length=267, #99
            stop_grow_count=20
        )),

    'Yearly': dict(
        model_parameters=dict(            
            min_length=32,
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=1
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=150,
            max_length=81, #99
            stop_grow_count=20
        ))
}

FFORMA_CONFIGS = {
    'Hourly': dict(
        n_estimators=2000,  # Number of iterations 100 default
        min_data_in_leaf=63,
        num_leaves=135,
        eta=0.607,  # learning rate
        max_depth=61,  # Max tree depth
        subsample=0.49,  # Bagging fraction (overfitting and speed) 1.0 default
        colsample_bytree=0.9),

    'Weekly': dict(
        n_estimators=2000,  # Number of iterations 100 default
        min_data_in_leaf=50,
        num_leaves=19,
        eta=0.46,  # learning rate
        max_depth=17,  # Max tree depth
        subsample=0.49,  # Bagging fraction (overfitting and speed) 1.0 default
        colsample_bytree=0.9),

    'Daily': dict(
        n_estimators=2000,  # Number of iterations 100 default
        min_data_in_leaf=200,
        num_leaves=94,
        eta=0.9,  # learning rate
        max_depth=9,  # Max tree depth
        subsample=0.52,  # Bagging fraction (overfitting and speed) 1.0 default
        colsample_bytree=0.49),

    'Monthly': dict(
        n_estimators=1200,  # Number of iterations 100 default
        min_data_in_leaf=100,  # Important for filter
        num_leaves=110,
        eta=0.2,  # learning rate
        max_depth=28,  # Max tree depth
        subsample=0.5,  # Bagging fraction (overfitting and speed) 1.0 default
        colsample_bytree=0.5),

    'Yearly': dict(
        n_estimators=1200,  # Number of iterations 100 default
        min_data_in_leaf=100,  # Important for filter
        num_leaves=110,
        eta=0.1,  # learning rate
        max_depth=28,  # Max tree depth
        subsample=0.5,  # Bagging fraction (overfitting and speed) 1.0 default
        colsample_bytree=0.5),

    'Quarterly': dict(
        n_estimators=2000,  # Number of iterations 100 default
        min_data_in_leaf=50,  # Important for filter
        num_leaves=94,
        eta=0.75,  # learning rate
        max_depth=43,  # Max tree depth
        subsample=0.81,  # Bagging fraction (overfitting and speed) 1.0 default
        colsample_bytree=0.49)
}

NEURALAVERAGE_CONFIGS = {
     'Hourly_2': dict(
        model_parameters=dict(
            layer_units=[80,20,6]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=2400,
            stop_grow_count=100
        )),

    'Weekly_2': dict(
        model_parameters=dict(
            layer_units=[80,20,6]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=2400,
            stop_grow_count=100
        )),
    
    'Daily_2': dict(
        model_parameters=dict(
            layer_units=[160,40,6]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=2400,
            stop_grow_count=100
        )),
    
    'Monthly_2': dict(
        model_parameters=dict(
            layer_units=[480,120,6]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=2400,
            stop_grow_count=100
        )),
    
    'Quarterly_2': dict(
        model_parameters=dict(
            layer_units=[320,80,6]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=2400,
            stop_grow_count=100
        )),

    'Yearly_2': dict(
        model_parameters=dict(
            layer_units=[320,80,6]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=2400,
            stop_grow_count=100
        )),

    'Hourly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=1200,
            epochs=600,
            stop_grow_count=15
        )),

    'Weekly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=300,
            stop_grow_count=3
        )),

    'Daily': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=52,
            epochs=300,
            stop_grow_count=3
        )),

    'Monthly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=52,
            epochs=300,
            stop_grow_count=3
        )),

    'Yearly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=52,
            epochs=300,
            stop_grow_count=3
        )),

    'Quarterly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=52,
            epochs=300,
            stop_grow_count=3
        ))
}

NEURALSTACK_CONFIGS = {
    'Hourly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=1200,
            epochs=600,
            stop_grow_count=15
        )),

    'Weekly': dict(
        model_parameters=dict(
            layer_units=[100, 100, 100, 100, 100, 50, 50, 50, 50, 20, 20]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=225,
            epochs=300,
            stop_grow_count=15
        )),

    'Daily': dict(
        model_parameters=dict(
            layer_units=[10, 10, 10]
        ),
        train_parameters=dict(
            learn_rate=3e-4,
            batch_size=225,
            epochs=300,
            stop_grow_count=15
        )),

    'Monthly': dict(
        model_parameters=dict(
            layer_units=[10, 10, 10]
        ),
        train_parameters=dict(
            learn_rate=3e-4,
            batch_size=225,
            epochs=300,
            stop_grow_count=15
        )),

    'Yearly': dict(
        model_parameters=dict(
            layer_units=[10, 10, 10]
        ),
        train_parameters=dict(
            learn_rate=3e-4,
            batch_size=225,
            epochs=300,
            stop_grow_count=15
        )),

    'Quarterly': dict(
        model_parameters=dict(
            layer_units=[10, 10, 10]
        ),
        train_parameters=dict(
            learn_rate=3e-4,
            batch_size=225,
            epochs=300,
            stop_grow_count=15
        ))
}

FEATURE_CONFIGS = {
    'Hourly': ['mf_hurst', 'mf_hw_alpha', 'mf_lumpiness', 'mf_seas_acf1'],

    'Weekly': ['mf_hurst', 'mf_stability', 'mf_spike', 'mf_e_acf1', 'mf_e_acf10', 'mf_lumpiness', 'mf_diff1x_pacf5',
               'mf_flat_spots',
               'mf_entropy', 'mf_x_acf10', 'mf_diff1_acf1', 'mf_diff2_acf1', 'mf_unitroot_kpss', 'mf_diff1_acf10'],

    'Daily': ['mf_series_length', 'mf_unitroot_kpss', 'mf_spike', 'mf_e_acf1', 'mf_e_acf10', 'mf_x_pacf5',
              'mf_nonlinearity',
              'mf_lumpiness', 'mf_flat_spots', 'mf_crossing_points', 'mf_arch_lm', 'mf_svd_entropy', 'mf_std',
              'mf_x_acf1', 'mf_x_acf10'],

    'Yearly': ['mf_hurst', 'mf_unitroot_pp', 'mf_unitroot_kpss', 'mf_crossing_points', 'mf_svd_entropy', 'mf_x_acf1',
               'mf_x_acf10', 'mf_diff1_acf1', 'mf_beta', 'mf_diff1_acf1', 'mf_diff1_acf10', 'mf_lumpiness',
               'mf_flat_spots'],

    'Monthly': ['mf_unitroot_pp', 'mf_unitroot_kpss', 'mf_hw_gamma', 'mf_trend', 'mf_spike', 'mf_seasonal_strength',
                'mf_lumpiness',
                'mf_crossing_points', 'mf_arch_lm', 'mf_svd_entropy', 'mf_linearity'],

    'Quarterly': ['mf_hurst', 'mf_series_length', 'mf_unitroot_pp', 'mf_unitroot_kpss', 'mf_hw_beta', 'mf_trend',
                  'mf_spike', 'mf_linearity', 'mf_diff1x_pacf5', 'mf_lumpiness',
                  'mf_crossing_points', 'mf_arch_lm', 'mf_svd_entropy', 'mf_seas_acf1', 'mf_x_acf1', 'mf_x_acf10']
}
