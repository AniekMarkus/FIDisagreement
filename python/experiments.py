
# Modules
import argparse
import importlib
# import Semantic_Meaningfulness_v2
# from Semantic_Meaningfulness_v2 import Sematic
# importlib.reload(Semantic_Meaningfulness_v2)

import joblib
import numpy as np
import pandas as pd
import torch
import random
import os
import pickle
import shutil
import re
from scipy.io import arff
from sklearn import metrics
import random
import xgboost

import torch
import torch.nn as nn

from pathlib import Path
from datetime import date

os.environ["CUDA_VISIBLE_DEVICES"]=""

# Get functions in other Python scripts
from help_functions import *
from simulate_data import *
from feature_importance import *
from models import *

import warnings
warnings.filterwarnings('ignore')

def setup(output_folder, clean):
    if clean and output_folder.exists():
        shutil.rmtree(output_folder)

    if not output_folder.exists():
                os.mkdir(output_folder)
                os.mkdir(output_folder / "data")
                os.mkdir(output_folder / "models")
                os.mkdir(output_folder / "feature_importance")
                os.mkdir(output_folder / "eval")


def get_data(data, repeat, root_folder, output_folder, rerun):
    data_folder = output_folder / "data"

    if rerun:
        # Process data
        if data in ['dgp1']:
            X, y = simulate_data(data, root_folder, settings_file='dgp_specs.csv')
        else:
            X, y = load_data(data, root_folder)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Save data
        X_train.to_csv(data_folder / str(f"{data}-{repeat}-Xtrain.csv"), index=False)
        pd.DataFrame(y_train).to_csv(data_folder / str(f"{data}-{repeat}-ytrain.csv"), index=False)

        X_test.to_csv(data_folder / str(f"{data}-{repeat}-Xtest.csv"), index=False)
        pd.DataFrame(y_test).to_csv(data_folder / str(f"{data}-{repeat}-ytest.csv"), index=False)

    else:
        # Load data
        X_train = pd.read_csv(data_folder / str(f"{data}-{repeat}-Xtrain.csv"))
        y_train = pd.read_csv(data_folder / str(f"{data}-{repeat}-ytrain.csv"))

        X_test = pd.read_csv(data_folder / str(f"{data}-{repeat}-Xtest.csv"))
        y_test = pd.read_csv(data_folder / str(f"{data}-{repeat}-ytest.csv"))

    print("get_data done")
    return X_train, X_test, y_train, y_test


def simulate_data(data, root_folder, settings_file):
    # Input
    input_settings = pd.read_csv(root_folder + '/input-settings/' + settings_file, index_col=0, sep=';')

    # Generate data using input parameters
    dgp = input_settings.loc[input_settings.name == data]

    # TODO: check this function
    [X, y] = simulate(N=dgp.N.values,
                      o=dgp.o.values,
                      beta=dgp.beta.values,
                      F=dgp.F.values,
                      inf=dgp.inf.values,
                      t=dgp.t.values,
                      rho=dgp.rho.values,
                      e=dgp.e.values,
                      A=dgp.A.values,
                      L=dgp.L.values,
                      seed=2023)
    return X, y

def load_data(data, root_folder):
    # TODO: give specs data format needs to satisfy

    # Input
    input_data = arff.loadarff(root_folder + "/input-data/" + data + ".arff")
    input_data = pd.DataFrame(input_data[0])

    # Split data
    X = input_data[input_data.columns[~input_data.columns.isin(['class'])]]
    y = input_data['class']

    return X, y

def modify_data(data, output_folder, params, X_train, X_test, y_train, y_test):
    data_folder = output_folder / "data"

    if params != '':
        # Modify data
        if params['change'] == "num_features":
            if params['method'] == "random":
                # Randomly select subset of features (change X data)
                selected_vars = random.sample(range(X_train.shape[1]), min(params["value"], X_train.shape[1]))
                X_train = X_train.iloc[:, selected_vars]
                X_test = X_test.iloc[:, selected_vars]

        elif params['change'] == "num_obs":
            if params['method'] == "random":
                # Randomly select subset of observations (change train set)
                selected_rows = random.sample(range(X_train.shape[0]), min(params["value"], X_train.shape[0]))
                X_train = X_train.iloc[selected_rows]
                y_train = y_train.iloc[selected_rows]

        elif params['change'] == "correlation":
            # Remove features with correlation above certain threshold (change X data)
            # TODO: implement scalable way to compute this
            cor_matrix = X_train.corr(method="pearson")

            # Remove variance on diagonal + upper/lower triangle and take absolute value
            diag_matrix = np.zeros((X_train.shape[1], X_train.shape[1]), int)
            np.fill_diagonal(diag_matrix, 1)
            cor_matrix = np.abs(cor_matrix - diag_matrix)

            # Find features above threshold and remove remaining features for further computation
            selected_vars = cor_matrix.index[(cor_matrix>params['value']).any()]
            X = X_train.loc[:, selected_vars]

            # Remove those one by one
            removed_vars = []

            while selected_vars.shape[0] > 0:
                f = selected_vars[-1]  # Starting with last, assuming less important
                X = X.drop(f, axis=1)
                removed_vars.append(f)

                # TODO: implement scalable way to compute this
                cor_matrix = X.corr(method="pearson")

                # Find features above threshold and remove remaining features for further computation
                selected_vars = cor_matrix.index[(cor_matrix>params['value']).any()]
                X = X.loc[:, selected_vars]

            # Get which features where removed
            # print(removed_vars)

            # Remove from X data
            X_train = X_train.drop(removed_vars, axis=1)
            # print(X_train.corr())
            X_test = X_test.drop(removed_vars, axis=1)

    # Save modified data
    X_train.to_csv(data_folder / str(f"{data}-Xtrain.csv"), index=False)
    pd.DataFrame(y_train).to_csv(data_folder / str(f"{data}-ytrain.csv"), index=False)

    X_test.to_csv(data_folder / str(f"{data}-Xtest.csv"), index=False)
    pd.DataFrame(y_test).to_csv(data_folder / str(f"{data}-ytest.csv"), index=False)

    print("> modify_data done")
    return X_train, X_test, y_train, y_test


def get_model(X_train, y_train, model, data, rerun):
    model_folder = output_folder / "models"

    if rerun:
        # Train
        ml_model, coef_model = globals()[f"fit_{model}"](X_train, y_train)

        # Save
        joblib.dump(ml_model, model_folder / str(f"{data}-{model}-model.pkl"))
        coef_model.to_csv(output_folder / "models" / str(f"{data}-{model}-coef.csv"), index=True)

    else:
        # Load
         file_name = model_folder / str(f"{data}-{model}-model.pkl")
         ml_model = joblib.load(file_name)
         coef_model = pd.read_csv(output_folder / "models" / str(f"{data}-{model}-coef.csv"))

    print("> get_model done")
    return ml_model, coef_model

def test_model(ml_model, coef_model, X_test, y_test, model, data):
    pred_prob = wrapper_predict(ml_model, X_test, prob=True)
    pred_class = np.where(pred_prob < 0.5, 0, 1)

    pred_outcomes = pred_class.sum()

    auc_score = metrics.roc_auc_score(y_test, pred_prob)
    auprc_score = metrics.average_precision_score(y_test, pred_prob)

    performance = pd.Series({'data': data,
                             'model': model,
                             'coef_non-zero': (coef_model.value != 0).sum(),
                             'pred_non-zero': pred_outcomes,
                             'perf_auc': auc_score,
                             'perf_auprc': auprc_score}, dtype='object')

    performance.to_csv(output_folder / "models" / str(f"{data}-{model}-perf.csv"), header=False)

    print("> test_data done")
    return performance

#      # TODO: compute for data folds??
#     # cross validation to compute robust feature importance
#     # kf = KFold(n_splits=folds)
#     # all_coefficients = 0
#
#     i = 1
#     # for incl_index, excl_index in kf.split(X_train):
#     print('Computing FI for fold: ' + str(i))
#     # X_subset, X_not = X_train.iloc[incl_index], X_train.iloc[excl_index]
#     # y_subset, y_not = y_train.iloc[incl_index], y_train.iloc[excl_index]
#     X_subset = X_train
#     y_subset = y_train

if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    i=None

    # SETTINGS
    root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"
    output_folder = Path(root_folder + "/results/output_" + str(date.today()))

    # TODO: different hyperparameter settings?
    # hyperparams={
    #     0: {"lr": 0.001, "epochs": 10, "batch_size": 16, "hidden_size": [18, 9, 3]}
    # }

    # SEED
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # PARAMETERS
    parser = argparse.ArgumentParser(
        prog = 'ExperimentsFI',
        description = 'Train model and compute feature importance',
        epilog = '...')

    parser.add_argument('data')
    parser.add_argument('model')
    parser.add_argument('fi_method')
    # parser.add_argument('metric', default='all')
    parser.add_argument('repeats', type=int, default=1)
    parser.add_argument('--modify-data', dest="modify_data", action='store_true')
    parser.set_defaults(modify_data=False)
    parser.add_argument('--use-model', dest='run_model', action='store_false') # If use existing model: don't run model
    parser.set_defaults(run_model=True)
    parser.add_argument('--use-data', dest='run_data', action='store_false') # If use existing data: don't run data
    parser.set_defaults(run_data=True)
    parser.add_argument('--clean', action='store_true')
    parser.set_defaults(clean=False)
    # parser.add_argument('save_path', default='./Results/')

    args = parser.parse_args()

    # If use existing model: don't run data
    if not args.run_model:
        args.run_data = False

    # Setings to use for data modification
    if args.modify_data:
        modify_params={
            0: {"change": "baseline"},
            10: {"change": "num_features", "value": 10, "method": "random"},
            11: {"change": "num_features", "value": 5, "method": "random"},
            20: {"change": "num_obs", "value": 1000, "method": "random"},
            21: {"change": "num_obs", "value": 500, "method": "random"},
            # 30: {"change": "inf_features", "value": 0.9, "method": "univariate"},
            # 31: {"change": "inf_features", "value": 0.7, "method": "univariate"},
            40: {"change": "correlation", "value": 0.9, "method": "univariate"},
            41: {"change": "correlation", "value": 0.7, "method": "univariate"},
            42: {"change": "correlation", "value": 0.5, "method": "univariate"},
            # 43: {"change": "correlation", "value": 0.3, "method": "univariate"}
            # TODO: check if no data left for modify_params
        }
        pd.DataFrame.from_dict(modify_params, orient="index").to_csv(output_folder / "data" / "modify_params.csv")

    else:
        modify_params={0: ''}

    print(f'Parameters : data = {args.data}, model = {args.model}, fi_method = {args.fi_method}, repeats = {args.repeats}, '
          f'modify_data = {args.modify_data}, run_model = {args.run_model}, run_data = {args.run_data}, clean = {args.clean}')

    # CREATE FOLDERS
    setup(output_folder, args.clean)

    # RUN ANALYSIS
    for repeat in range(1, args.repeats + 1):
        # Data
        X_train, X_test, y_train, y_test = get_data(args.data, repeat, root_folder, output_folder, args.run_data)

        for p in modify_params.keys():
            data = str(f"{args.data}-{repeat}-v{p}")

            # TODO: avoid running modify data again if not necessary
            X_train_p, X_test_p, y_train_p, y_test_p = modify_data(data, output_folder, modify_params[p], X_train, X_test, y_train, y_test)

            # Get ML model
            ml_model, coef_model = get_model(X_train_p, y_train_p, args.model, data, args.run_model)
            performance = test_model(ml_model, coef_model, X_test_p, y_test_p, args.model, data)

            # Compute FI
            fi_values, fi_time = locals()[f"{args.fi_method}"](ml_model, X_train_p, y_train_p)
            print("> fi_method done")
            fi_values = pd.DataFrame(np.array(fi_values), index=X_train_p.columns.values, columns=['value'])
            fi_values.to_csv(output_folder / "feature_importance" / str(f"{data}-{args.model}-{args.fi_method}-fi_values.csv"))

            # Save results of run
            performance = performance.to_frame()

            summary=pd.DataFrame([])
            summary['repeat']=[f'{repeat}']
            summary['fi_method']=[f'{args.fi_method}']
            summary['fi_zero']=sum(fi_values.value == 0)
            summary['fi_non-zero']=sum(fi_values.value != 0)
            summary['fi_time'] = fi_time
            summary = summary.transpose()

            summary = pd.concat([performance, summary], axis=0)
            summary.to_csv(f'{output_folder}/eval/{data}-{args.model}-{args.fi_method}-summary.csv', header=False)
            print("> saved summary results")