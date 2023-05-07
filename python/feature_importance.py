# Modules
import numpy as np
import pandas as pd
import os

import shap
import sage
import time
from functools import partial
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Get functions in other Python scripts
from help_functions import *
from models import *
from sage_utils import Surrogate, MaskLayer1d, KLDivLoss

# TODO: set (changing) seed in shap methods?

### WRAPPER FUNCTIONS
def permutation_auc(model, X, y, data, convergence=False):
    if convergence:
        wrapper_fi = partial(compute_permutation_custom, model, X, y, 'roc_auc') # model, X, y, score used as arguments
        fi_values, elapsed_time = check_convergence(wrapper_fi, 1, 1)
    else:
        fi_values, elapsed_time = compute_permutation_custom(model, X, y, scoring='roc_auc', repeat=10)

    return fi_values, elapsed_time

def permutation_mse(model, X, y, data, convergence=False):
    if convergence:
        wrapper_fi = partial(compute_permutation_custom, model, X, y, 'neg_mean_squared_error') # model, X, y, score used as arguments
        fi_values, elapsed_time = check_convergence(wrapper_fi, 1, 1)
    else:
        fi_values, elapsed_time = compute_permutation_custom(model, X, y, scoring='neg_mean_squared_error', repeat=10)

    return fi_values, elapsed_time

def permutation_ba(model, X, y, data, convergence=False):
    threshold = get_threshold(data)
    if convergence:
        wrapper_fi = partial(compute_permutation_custom, model, X, y, 'balanced_accuracy', threshold)  # model, X, y, score used as arguments
        fi_values, elapsed_time = check_convergence(wrapper_fi, 1, 1)
    else:
        fi_values, elapsed_time = compute_permutation_custom(model, X, y, scoring='balanced_accuracy', threshold=threshold, repeat=10)

    return fi_values, elapsed_time


def kernelshap(model, X, y, data, convergence=False):
    if convergence:
        wrapper_fi = partial(compute_kernelshap, model, X)  # model, X used as arguments
        fi_values, elapsed_time = check_convergence(wrapper_fi, 250, 250)
    else:
        fi_values, elapsed_time = compute_kernelshap(model, X, samples=3000) # TODO: change default

    return fi_values, elapsed_time


def sage_marginal(model, X, y, data, convergence=False):
    if convergence:
        wrapper_fi = partial(compute_sage, model, X, y, 'marginal')  # model, X, y used as arguments
        fi_values, elapsed_time = check_convergence(wrapper_fi, 250, 250)
    else:
        fi_values, elapsed_time = compute_sage(model, X, y, removal='marginal', samples=3000) # TODO: change default

    return fi_values, elapsed_time


def sage_conditional(model, X, y, data, convergence=False): # TODO: check implementations
    if convergence:
        wrapper_fi = partial(compute_sage, model, X, y, 'surrogate')  # model, X, y used as arguments
        fi_values, elapsed_time = check_convergence(wrapper_fi, 250, 250)
    else:
        fi_values, elapsed_time = compute_sage(model, X, y, removal='surrogate', samples=3000) # TODO: change default

    return fi_values, elapsed_time


def loco_auc(model, X, y, data):
    return compute_loco_custom(model, X, y, scoring='roc_auc')

def loco_mse(model, X, y, data):
    return compute_loco_custom(model, X, y, scoring='neg_mean_squared_error')

def loco_ba(model, X, y, data):
    threshold = get_threshold(data)
    return compute_loco_custom(model, X, y, scoring='balanced_accuracy', threshold=threshold)



def check_convergence(wrapper_fi, start, step, stop=0.025):
    # Start values
    fi_values, elapsed_time = wrapper_fi(start)
    value=start+step
    converged=False

    while not converged:
        fi_values_new, elapsed_time_new = wrapper_fi(value)

        dist = np.linalg.norm(fi_values-fi_values_new)

        # Check if L2 distance smaller than stop
        if dist < stop:
            print("Converged at " + str(value) + " with distance " + str(dist))
            converged=True
        else:
            print("Distance at " + str(value) + " is " + str(dist))

            # Update values for next round
            value=value+step
            fi_values = fi_values_new

    return fi_values_new, elapsed_time_new



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


def compute_permutation_custom(ml_model, X, y, scoring='roc_auc', threshold=0.5, repeat=1):
    print("Busy with permutation custom FI")

    start_time = time.time()

    if scoring == 'roc_auc':
        pred = wrapper_predict(ml_model, X)
        perf_full_mod = roc_auc_score(y, pred)
    elif scoring == 'neg_mean_squared_error':
        # pred = ml_model.predict_proba(X.values)[:, 1]
        pred = wrapper_predict(ml_model, X)
        perf_full_mod = -1 * mean_squared_error(y, pred, squared=False)  # RMSE
    elif scoring == 'balanced_accuracy':
        pred = wrapper_predict(ml_model, X, prob=False, threshold=threshold)
        perf_full_mod = balanced_accuracy_score(y, pred)

    # Initialize a list of results
    results = []
    # Iterate through each predictor
    for predictor in X: # predictor = X.columns[0]

        sum_new_perf = 0

        for r in range(1, repeat + 1):
            # Create a copy of X_test
            X_temp = X.copy()

            # Scramble the values of the given predictor
            X_temp[predictor] = X[predictor].sample(frac=1).values

            # Calculate the new performance
            if scoring == 'roc_auc':
                pred_temp = wrapper_predict(ml_model, X_temp)
                new_perf = roc_auc_score(y, pred_temp)
            elif scoring == 'neg_mean_squared_error':
                pred_temp = wrapper_predict(ml_model, X_temp)
                new_perf = -1 * mean_squared_error(y, pred_temp, squared=False)  # RMSE
            elif scoring == 'balanced_accuracy':
                pred_temp = wrapper_predict(ml_model, X_temp, prob=False)
                new_perf = balanced_accuracy_score(y, pred_temp)

            sum_new_perf = sum_new_perf + new_perf

        new_perf_avg = sum_new_perf / repeat

        # Append the decrease in perf to the list of results
        results.append({'pred': predictor,
                        'score': perf_full_mod - new_perf_avg})

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

    # X_summary = shap.kmeans(X, 50)
    X_summary = shap.sample(X, samples)

    wrapper_predict_model = partial(wrapper_predict, model)  # model is used as first argument -> ml_model
    explainer = shap.KernelExplainer(wrapper_predict_model, X_summary)

    kernelshap_values = explainer.shap_values(X_summary, l1_reg='num_features(10)')

    if model.name == "nn": # this model outputs a list (of 1 array)
        kernelshap_values = kernelshap_values[0]

    # visualize the first prediction's explanation
    # shap.plots.waterfall(kernelshap_values[0])

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time kernelshap FI: ", elapsed_time)

    return kernelshap_values.mean(axis=0), elapsed_time


def compute_sage(model, X, y, removal='marginal', samples=1000, binary_outcome=True):
    print("Busy with sage FI")

    start_time = time.time()

    # Convert list to array
    X = X.values
    y = np.array(y)

    if removal == 'marginal':
        # Set up an imputer to handle missing features
        X_summary = shap.sample(X, samples)

        # Marginalizing out removed features with their marginal distribution
        imputer = sage.MarginalImputer(model, X_summary) # TODO: check how predict wrapper works here

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

        X_surr = torch.tensor(X_surr, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

        data_surr = torch.utils.data.TensorDataset(X_surr)
        data_val = torch.utils.data.TensorDataset(X_val)

        surrogate.train_original_model(
            train_data=data_surr,
            val_data=data_val,
            original_model=model,
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
    # number = X.shape[1]*1000
    # sage_values = estimator(X, y, verbose=True, bar=True, detect_convergence=False, n_permutations=number)
    sage_values = estimator(X, y, detect_convergence=True)
    # sage_values.plot(feature_names)

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time sage FI: ", elapsed_time)

    # feature_names = X_train.columns.tolist()[:-1]
    # sage_values.plot(feature_names, title='Feature Importance (Surrogate)')

    return sage_values.values, elapsed_time


def compute_loco_custom(ml_model, X, y, scoring='roc_auc', threshold=0.5):
    print("Busy with LOCO custom FI")

    start_time = time.time()

    if scoring == 'roc_auc':
        pred = wrapper_predict(ml_model, X)
        perf_full_mod = roc_auc_score(y, pred)
    elif scoring == 'neg_mean_squared_error':
        pred = wrapper_predict(ml_model, X)
        perf_full_mod = -1 * mean_squared_error(y, pred, squared=False)  # RMSE
    elif scoring == 'balanced_accuracy':
        pred = wrapper_predict(ml_model, X, prob=False, threshold=threshold)
        perf_full_mod = balanced_accuracy_score(y, pred)

    # Initialize a list of results
    results = []
    # Iterate through each predictor
    for predictor in X:  # predictor = X.columns[0]

        # Create a copy of X_test
        X_temp = X.copy()

        # Remove the given predictor
        X_temp.drop(predictor, axis=1, inplace=True)

        # Retrain model
        ml_model_temp, coef_model_temp = get_model(X_temp, y, ml_model.name)

        # Calculate the new performance
        if scoring == 'roc_auc':
            pred_temp = wrapper_predict(ml_model_temp, X_temp)
            new_perf = roc_auc_score(y, pred_temp)
        elif scoring == 'neg_mean_squared_error':
            pred_temp = wrapper_predict(ml_model_temp, X_temp)
            new_perf = -1 * mean_squared_error(y, pred_temp, squared=False)  # RMSE
        elif scoring == 'balanced_accuracy':
            pred_temp = wrapper_predict(ml_model_temp, X_temp, prob=False)
            new_perf = balanced_accuracy_score(y, pred_temp)

        # Append the decrease in perf to the list of results
        results.append({'pred': predictor,
                        'score': perf_full_mod - new_perf})

    # Convert to a pandas dataframe and rank the predictors by score
    resultsdf = pd.DataFrame(results)

    loco_values = resultsdf.score

    end_time = time.time()
    elapsed_time = end_time-start_time
    # print("Time LOCO FI: ", elapsed_time)

    return loco_values, elapsed_time
