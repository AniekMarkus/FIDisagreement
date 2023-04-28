
import shap
import sage
import numpy as np
import pandas as pd
import sklearn
import time

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models import *

import torch
import torch.nn as nn
from sage_utils import Surrogate, MaskLayer1d, KLDivLoss

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO: set (changing) seed in shap methods?

### WRAPPER FUNCTIONS
# TODO: check stability results/ different values per run?

def permutation_auc(model, X, y):
    # TODO:choose which one to use compute_permutation(model, X_subset, y_subset, scoring = 'roc_auc')
    return compute_permutation_custom(model, X, y, scoring='roc_auc')

def permutation_mse(model, X, y):
    return compute_permutation_custom(model, X, y, scoring='neg_mean_squared_error')

def permutation_ba(model, X, y):
    return compute_permutation_custom(model, X, y, scoring='balanced_accuracy')

# TODO: check implementations SHAP
def kernelshap_500(model, X, y):
    return compute_kernelshap(model, X, samples=500)

def shap_marginal500(model, X, y):
    return  compute_sage(model, X, y, samples=500, removal='marginal', binary_outcome=True)

def shap_conditional(model, X, y):
    return compute_sage(model, X, y, removal='surrogate', binary_outcome=True)


### HELP FUNCTIONS
def compute_permutation(model, X, y, scoring='roc_auc'):
    print("Busy with permutation FI")
    start_time = time.time()

    res = permutation_importance(model, X.values, y, n_repeats=5, random_state=0, scoring=scoring)
    permutation_values = res.importances_mean

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time permutation FI: ", elapsed_time)

    return permutation_values, elapsed_time


def compute_permutation_custom(model, X, y, scoring='roc_auc'):
    print("Busy with permutation custom FI")

    start_time = time.time()

    if scoring == 'roc_auc':
        perf_full_mod = roc_auc_score(y, model.predict_proba(X.values)[:, 1])
    elif scoring == 'neg_mean_squared_error':
        perf_full_mod = -1 * mean_squared_error(y, model.predict_proba(X.values)[:, 1], squared=False)  # RMSE
    elif scoring == 'balanced_accuracy':
        perf_full_mod = balanced_accuracy_score(y, (model.predict_proba(X.values)[:, 1] >= 0.1).astype(int))

    # Initialize a list of results
    results = []
    # Iterate through each predictor
    for predictor in X:  # predictor = X.columns[0]
        # print(predictor)

        # Create a copy of X_test
        X_temp = X.copy()

        # Scramble the values of the given predictor
        X_temp[predictor] = X[predictor].sample(frac=1).values

        # Calculate the new RMSE
        # new_perf = mean_squared_error(model.predict(X_temp.values), y, squared=False)
        # new_perf = balanced_accuracy_score(model.predict(X_temp.values),  y.iloc[:, 0])
        # TODO: check warning y_pred contains classes not in y_true

        if scoring == 'roc_auc':
            new_perf = roc_auc_score(y, model.predict_proba(X_temp.values)[:, 1])
        elif scoring == 'neg_mean_squared_error':
            new_perf = -1 * mean_squared_error(y, model.predict_proba(X_temp.values)[:, 1], squared=False)  # RMSE
        elif scoring == 'balanced_accuracy':
            new_perf = balanced_accuracy_score(y,
                                               (model.predict_proba(X_temp.values)[:, 1] >= 0.1).astype(int))

        # Append the decrease in perf to the list of results
        results.append({'pred': predictor,
                        'score': perf_full_mod - new_perf})

    # Convert to a pandas dataframe and rank the predictors by score
    resultsdf = pd.DataFrame(results)

    permutation_values = resultsdf.score

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time permutation FI: ", elapsed_time)

    return permutation_values, elapsed_time


def compute_shap(model, X): # TODO: nothing changes with link function?
    print("Busy with shap FI")

    start_time = time.time()
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

    explainer = shap.Explainer(model, X, link=shap.links.identity)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    # shap.plots.waterfall(shap_values[0])

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time shap FI: ", elapsed_time)

    return shap_values, elapsed_time


def compute_linearshap(model, X):
    print("Busy with linearshap FI")

    start_time = time.time()

    explainer = shap.LinearExplainer(model, X, link=shap.links.identity)
    linearshap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation
    # shap.plots.waterfall(kernelshap_values[0])

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time linearshap FI: ", elapsed_time)

    return linearshap_values, elapsed_time


def compute_kernelshap(model, X, samples=1000):
    print("Busy with kernelshap FI")

    start_time = time.time()

    # Convert list to array
    # y = np.array(y)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

    # X_summary = shap.kmeans(X, 50)
    X_summary = shap.sample(X, samples)
    # explainer = shap.KernelExplainer(model.predict, X_summary)
    explainer = shap.KernelExplainer(model.predict_proba, X_summary)

    # kernelshap_values = explainer.shap_values(X_summary)[1]
    # kernelshap_values = explainer.shap_values(X_summary, l1_reg='num_features(10)')[1]
    kernelshap_values = explainer.shap_values(X_summary, nsamples=10 * X.shape[1] + 2048, l1_reg='num_features(10)')[1]

    # visualize the first prediction's explanation
    # shap.plots.waterfall(kernelshap_values[0])

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time kernelshap FI: ", elapsed_time)

    return kernelshap_values, elapsed_time


def compute_sage(model, X, y, samples=1000, removal='marginal', binary_outcome=True):
    print("Busy with sage FI")

    start_time = time.time()

    # Convert list to array
    X = X.values
    y = np.array(y)

    if removal == 'marginal':
        # Set up an imputer to handle missing features
        X_summary = shap.sample(X, samples)

        # Marginalizing out removed features with their marginal distribution
        imputer = sage.MarginalImputer(model, X_summary)

    elif removal == 'surrogate':
        # Conditional distribution inspired from https://github.com/iancovert/fastshap/blob/main/notebooks/census.ipynb
        surrogate_start = time.time()
        # device = torch.device('cuda')
        device = torch.device("cpu")

        # Create surrogate model
        num_features = X.shape[1]
        X_surr, X_val, y_surr, y_val = train_test_split(X, y, test_size=0.25, random_state=2022)

        surr = nn.Sequential(
                MaskLayer1d(value=0, append=True),
                nn.Linear(2 * num_features, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 2)).to(device)

        surrogate = Surrogate(surr, num_features)

        surrogate.train(
            train_data=(X_surr, y_surr),
            val_data=(X_val, y_val),
            batch_size=64,
            max_epochs=100,
            loss_fn=KLDivLoss(),
            validation_samples=10,
            validation_batch_size=10000,
            verbose=False)

        class Imputer:
            def __init__(self):
                self.num_groups = num_features

            def __call__(self, x, S):
                x = torch.tensor(x, dtype=torch.float32, device=device)
                S = torch.tensor(S, dtype=torch.float32, device=device)
                pred = surrogate(x, S).softmax(dim=-1)
                return pred.cpu().data.numpy()

                # Call surrogate model (with data normalization)
                # return surrogate(
                #     (torch.tensor((x - mean) / std, dtype=torch.float32, device=device),
                #      torch.tensor(S, dtype=torch.float32, device=device))
                # ).softmax(dim=1).cpu().data.numpy()

        imputer = Imputer()

        surrogate_end = time.time()
        surrogate_elapsed = surrogate_end - surrogate_start
        print("Time (in sec) to train surrogate model: " + str(surrogate_elapsed))
    else:
        raise ValueError("Removal method not implemented.")

    # Set up an estimator
    if binary_outcome:
        # estimator = sage.KernelEstimator(imputer, loss='cross entropy')
        estimator = sage.PermutationEstimator(imputer, loss='cross entropy')
        # Estimate SAGE values by unrolling permutations of feature indices
        # Changed np.max to np.nanmax in PermutationEstimator
    else:
        # estimator = sage.KernelEstimator(imputer, loss='mse')
        estimator = sage.PermutationEstimator(imputer, loss='mse')
        # TODO: test difference kernel and permutation estimator?

    # Calculate SAGE values
    number = X.shape[1]*1000
    sage_values = estimator(X, y, verbose=True, bar=True, detect_convergence=False, n_permutations=number)
    # sage_values = estimator(X, y, verbose=True, bar=True, detect_convergence=True)
    # sage_values.plot(feature_names)

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time sage FI: ", elapsed_time)

    # feature_names = X_train.columns.tolist()[:-1]
    # sage_values.plot(feature_names, title='Feature Importance (Surrogate)')

    return sage_values, elapsed_time