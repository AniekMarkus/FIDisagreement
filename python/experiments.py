
# Modules
import numpy as np
import pandas as pd
import os

import argparse
import importlib
import torch
import random
import re

from pathlib import Path
from datetime import date

os.environ["CUDA_VISIBLE_DEVICES"]=""

# Get functions in other Python scripts
from help_functions import *
from data import *
from models import *
from feature_importance import *

import warnings
warnings.filterwarnings('ignore')

if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    i=None

    # SETTINGS
    root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"

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
    parser.add_argument('folder', default=None)
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

    # If no folder given, use output_folder
    if args.folder is None:
        output_folder = Path(root_folder + "/results/output_" + str(date.today()))
    else:
        output_folder = Path(root_folder + "/results/" + str(args.folder))

    # If use existing model: don't run data
    if not args.run_model:
        args.run_data = False

    # CREATE FOLDERS
    setup(output_folder, args.clean)

    # Settings to use for data modification
    if args.modify_data:
        modify_params={
            0: {"change": "baseline"},
            10: {"change": "num_features", "value": 50, "method": "random"},
            11: {"change": "num_features", "value": 25, "method": "random"},
            12: {"change": "num_features", "value": 15, "method": "random"},
            13: {"change": "num_features", "value": 10, "method": "random"},
            # 14: {"change": "num_features", "value": 7, "method": "random"},
            # 15: {"change": "num_features", "value": 6, "method": "random"},
            16: {"change": "num_features", "value": 5, "method": "random"},
            # 17: {"change": "num_features", "value": 3, "method": "random"},
            20: {"change": "num_obs", "value": 5000, "method": "random"},
            # 21: {"change": "num_obs", "value": 2500, "method": "random"},
            22: {"change": "num_obs", "value": 1500, "method": "random"},
            23: {"change": "num_obs", "value": 750, "method": "random"},
            24: {"change": "num_obs", "value": 500, "method": "random"},
            # 25: {"change": "num_obs", "value": 250, "method": "random"},
            30: {"change": "num_outcomes", "value": 1000, "method": "random"},
            31: {"change": "num_outcomes", "value": 500, "method": "random"},
            32: {"change": "num_outcomes", "value": 250, "method": "random"},
            33: {"change": "num_outcomes", "value": 100, "method": "random"},
            40: {"change": "correlation", "value": 0.9, "method": "univariate"},
            41: {"change": "correlation", "value": 0.7, "method": "univariate"},
            42: {"change": "correlation", "value": 0.5, "method": "univariate"},
            # 43: {"change": "correlation", "value": 0.3, "method": "univariate"},
            50: {"change": "prev_features", "value": 0.05, "method": "random"},
            51: {"change": "prev_features", "value": 0.10526, "method": "random"}, # cumulative removed = 0.15 -> (0.95-0.85)/0.95
            52: {"change": "prev_features", "value": 0.11765, "method": "random"}  # cumulative removed = 0.25 -> (0.85-0.75)/0.85
        }
        pd.DataFrame.from_dict(modify_params, orient="index").to_csv(output_folder / "data" / "modify_params.csv")

    else:
        modify_params={0: ''}

    print(f'Parameters : data = {args.data}, model = {args.model}, fi_method = {args.fi_method}, repeats = {args.repeats}, '
          f'modify_data = {args.modify_data}, run_model = {args.run_model}, run_data = {args.run_data}, clean = {args.clean}')

    # RUN ANALYSIS
    for repeat in range(1, args.repeats + 1):
        # Data
        X_train, X_test, y_train, y_test = get_data(args.data, repeat, root_folder, output_folder, args.run_data)

        for p in modify_params.keys():
            data = str(f"{args.data}-{repeat}-v{p}")

            X_train, X_test, y_train, y_test, change = modify_data(data, output_folder, modify_params[p], X_train, X_test, y_train, y_test, args.run_data)

            # Do not run remaining analysis if data empty or no change after modification
            if (X_train.shape[0] > 0 and X_train.shape[1] > 0) and change:
                # Get ML model
                ml_model, coef_model = get_model(X_train, y_train, args.model, data, output_folder, args.run_model)
                performance = test_model(output_folder, ml_model, coef_model, X_test, y_test, args.model, data)

                # Compute FI
                fi_values, fi_time = locals()[f"{args.fi_method}"](ml_model, X_train, y_train, data)
                print("> fi_method done")
                fi_values = pd.DataFrame(np.array(fi_values), index=X_train.columns.values, columns=['value'])
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