
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
from data import *

import warnings
warnings.filterwarnings('ignore')


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
            11: {"change": "num_features", "value": 0, "method": "random"},
            20: {"change": "num_obs", "value": 1000, "method": "random"},
            21: {"change": "num_obs", "value": 500, "method": "random"},
            # 30: {"change": "inf_features", "value": 0.9, "method": "univariate"},
            # 31: {"change": "inf_features", "value": 0.7, "method": "univariate"},
            40: {"change": "correlation", "value": 0.9, "method": "univariate"},
            41: {"change": "correlation", "value": 0.7, "method": "univariate"},
            42: {"change": "correlation", "value": 0.5, "method": "univariate"},
            # 43: {"change": "correlation", "value": 0.3, "method": "univariate"}
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

            # Do not run remaining analysis if data empty after modification
            if X_train_p.shape[0] > 0 and X_train_p.shape[1] > 0:
                # Get ML model
                ml_model, coef_model = get_model(output_folder, X_train_p, y_train_p, args.model, data, args.run_model)
                performance = test_model(output_folder, ml_model, coef_model, X_test_p, y_test_p, args.model, data)

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