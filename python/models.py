
import scipy.stats
import xgboost
import sklearn
import scipy
import numpy as np

def fit_linear(X_train, y_train):
    model = sklearn.linear_model.LinearRegression(random_state=2022)
    model.fit(X_train.values, y_train.values)
    coef = model.coef_

    return model, coef


def fit_logistic(X_train, y_train):
    # TODO: check warning "A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel()."

    model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1, solver='saga', random_state=2022, max_iter=100000)
    # model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1, random_state=2022, max_iter=100000)
    model.fit(X_train.values, y_train.values)

    # z = X_train.dot(model.coef_.transpose())
    # coef = model.coef_*scipy.stats.logistic.pdf(z).mean() # TODO: check this!

    coef = model.coef_

    return model, coef

def fit_xgboost(X_train, X_test, y_train, y_test):
    # Train an XGBoost model with early stopping
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    model = xgboost.train(
        {"eta": 0.001, "subsample": 0.5, "max_depth": 2, "objective": "reg:logistic"}, dtrain, num_boost_round=200000,
        evals=((dtest, "test"),), early_stopping_rounds=20, verbose_eval=False
    )
    return model
