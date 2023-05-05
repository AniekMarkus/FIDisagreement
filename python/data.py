
# Modules
import importlib
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

# Get functions in other Python scripts


os.environ["CUDA_VISIBLE_DEVICES"]=""

# Get functions in other Python scripts
from simulate_data import *
from models import *

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

            # Remove from X data
            X_train = X_train.drop(removed_vars, axis=1)
            X_test = X_test.drop(removed_vars, axis=1)

    # Save modified data
    X_train.to_csv(data_folder / str(f"{data}-Xtrain.csv"), index=False)
    pd.DataFrame(y_train).to_csv(data_folder / str(f"{data}-ytrain.csv"), index=False)

    X_test.to_csv(data_folder / str(f"{data}-Xtest.csv"), index=False)
    pd.DataFrame(y_test).to_csv(data_folder / str(f"{data}-ytest.csv"), index=False)

    print("> modify_data done")
    return X_train, X_test, y_train, y_test

