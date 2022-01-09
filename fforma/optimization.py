from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from fforma import FFORMA

def get_optimal_params(train_errors, train_feats, test_feats, n_calls):
    # Bayesian optimisation dimsensions
    dimensions = [
        Integer(low=50, high=200, name='min_data_in_leaf'),
        Integer(low=10, high=150, name='num_leaves'),
        Real(low=0.05, high=0.9, name='eta'),
        Integer(low=8, high=64, name='max_depth'),
        Real(low=0.49, high=0.9, name='subsample'),
        Real(low=0.49, high=0.9, name='colsample_bytree'),
    ]

    default_parameters = [100, 80, 0.58, 14, 0.85, 0.77]

    @use_named_args(dimensions=dimensions)
    def _opt_model(min_data_in_leaf, num_leaves, eta, max_depth, subsample, colsample_bytree):

        parameters = dict(n_estimators=2000,  # Number of iterations 100 default
                          min_data_in_leaf=min_data_in_leaf,
                          num_leaves=num_leaves,
                          eta=eta,  # learning rate
                          max_depth=max_depth,  # Max tree depth
                          subsample=subsample,  # Bagging fraction (overfitting and speed) 1.0 default
                          colsample_bytree=colsample_bytree)
        fforma = FFORMA(params=parameters, verbose_eval=1)
        valid_best_score = fforma.fit(errors=train_errors, holdout_feats=train_feats, feats=test_feats)
        return valid_best_score

    # Find model hyperparameters
    searched_params = gp_minimize(_opt_model,
                                  dimensions,
                                  n_calls=n_calls,
                                  x0=default_parameters,
                                  verbose=True)

    # Create the model
    return searched_params