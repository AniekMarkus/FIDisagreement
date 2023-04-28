
import scipy.stats
import xgboost
import sklearn
import scipy
import numpy as np


def fit_xgboost(X_train, X_test, y_train, y_test):
    # Train an XGBoost model with early stopping
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    model = xgboost.train(
        {"eta": 0.001, "subsample": 0.5, "max_depth": 2, "objective": "reg:logistic"}, dtrain, num_boost_round=200000,
        evals=((dtest, "test"),), early_stopping_rounds=20, verbose_eval=False
    )
    return model
