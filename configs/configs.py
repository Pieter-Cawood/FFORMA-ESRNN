def alllcm(a,b,max):
    for i in range(min,max):
        if i%a==0 and i%b==0:
            print(i)

# alllcm(24,(2**5),24*10)

DEEPFFORMA_CONFIGS = {
    #minimum points you want available at the end from the shortest ts    
    #length-halvings-points: 320-5-0, 160-4-0, 80-3-0, 40-2-0, 20-1-0
    #
    #length-halvings-points: 288-5-9, 144-4-9, 72-3-9, 36-2-9, 18-1-9
    #length-halvings-points: 256-5-8, 128-4-8, 64-3-8, 32-2-8, 16-1-8
    #length-halvings-points: 224-5-7  112-4-7, 56-3-7, 28-2-7, 14-1-7
    #length-halvings-points: 192-5-6   96-4-6, 48-3-6, 24-2-6, 12-1-6    #
    #length-halvings-points: 160-5-5   80-4-5, 40-3-5, 20-2-5, 10-1-5    #     
    #length-halvings-points: 128-5-4   64-4-4, 32-3-4, 16-2-4,  8-1-4

    #length-halvings-points:  96-5-3   48-4-3, 24-3-3, 12-2-3,  6-1-3
    #length-halvings-points:  64-5-2   32-4-2, 16-3-2,  8-2-2,  4-1-2
    #
    #11-352, 12-384, 13-416, 14-448, 15-480, 16-512, 17-544, 18-576, 
    #19-608, 20-640, 21-672, 22-704, 23-736, 24-768,  27-864, 28-896
    #34-1088 35-1120
    'Hourly': dict(
        model_parameters=dict(
            adaptings=7,
            halvings=5, # 700 700
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=24 #[168,24]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=1000,
            max_length=256+24-1, #512, #960,
            stop_grow_count=100,
            augment=False
        )),
    
    'Daily': dict(
        model_parameters=dict(
            adaptings=5,
            halvings=5, #93 111
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=7, #[365,7]
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=250,
            max_length=256+7-1, #2940, #99-4315, 50-2940
            stop_grow_count=40,
            augment=False
        )),
    
    'Weekly': dict(
        model_parameters=dict(
            adaptings=7, #3
            halvings=5, #80 80
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=52
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=1000,
            max_length=256+52-1, #288 #99-2283
            stop_grow_count=100,
            augment=False
        )),
    
    'Monthly': dict(
        model_parameters=dict(
            adaptings=6,
            halvings=5, #42 66
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=12
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=150,
            max_length=256+12-1, #99-664
            stop_grow_count=20,
            augment=False
        )),
    
    'Quarterly': dict(
        model_parameters=dict(
            adaptings=6,
            halvings=5, #16, 24
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=4
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=150,
            max_length=256+4-1, #99-267
            stop_grow_count=20,
            augment=False
        )),
    
    'Yearly': dict(
        model_parameters=dict(
            adaptings=8,
            halvings=5, #13 13
            vgg_filters=None,
            res_filters=64,
            dropout_rate=0.1,
            seasons=1
        ),
        train_parameters=dict(
            learn_rate=1e-4,
            batch_size=92,
            epochs=150,
            max_length=256+1-1, #99-81
            stop_grow_count=20,
            augment=False
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
