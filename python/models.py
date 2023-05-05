
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

import warnings
warnings.filterwarnings('ignore')

def get_model(X_train, y_train, model, data=None, output_folder=None, rerun=True):

    if rerun:
        # Train
        ml_model, coef_model = globals()[f"fit_{model}"](X_train, y_train)

        if output_folder is not None:
            # Save
            joblib.dump(ml_model, output_folder / "models" / str(f"{data}-{model}-model.pkl"))
            coef_model.to_csv(output_folder / "models" / str(f"{data}-{model}-coef.csv"), index=True)

    else:
        # Load
        file_name = output_folder / "models" / str(f"{data}-{model}-model.pkl")
        ml_model = joblib.load(file_name)
        coef_model = pd.read_csv(output_folder / "models" / str(f"{data}-{model}-coef.csv"))

    print("> get_model done")
    return ml_model, coef_model

def fit_linear(X_train, y_train):
    ml_model = sklearn.linear_model.LinearRegression(random_state=2022)
    ml_model.fit(X_train.values, y_train.values)

    coef = pd.DataFrame(np.transpose(ml_model.coef_), index=X_train.columns.values, columns=['value'])

    return ml_model, coef


def fit_logistic(X_train, y_train):
    ml_model = sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1, solver='saga', random_state=2022, max_iter=100000)
    # model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1, random_state=2022, max_iter=100000)
    ml_model.fit(X_train.values, y_train.values)

    # z = X_train.dot(model.coef_.transpose())
    # coef = model.coef_*scipy.stats.logistic.pdf(z).mean() # TODO: check this!

    coef = pd.DataFrame(np.transpose(ml_model.coef_), index=X_train.columns.values, columns=['value'])

    ml_model.name = "logistic"
    return ml_model, coef


def fit_xgboost(X_train, y_train):
    # train an XGBoost model with early stopping
    X_input = X_train.values
    data = xgboost.DMatrix(X_input, label=y_train)

    ml_model = xgboost.train(
        {"eta": 0.001, "subsample": 0.5, "max_depth": 2, "objective": "binary:logistic"},
        data, num_boost_round=200000, evals=((data, "train"),),
        early_stopping_rounds=20, verbose_eval=False)

    coef = pd.DataFrame(0, index=X_train.columns.values, columns=['value']) # TODO: add value?

    ml_model.name = "xgboost"
    return ml_model, coef

def fit_nn(X_train, y_train):
    # TODO: check hyperparameters
    n_input, n_hidden, n_out, batch_size, learning_rate = X_train.shape[1], 15, 1, 100, 0.01

    data_x = torch.tensor(X_train.values, dtype=torch.float32)
    data_y = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)

    ml_model = nn.Sequential(nn.Linear(n_input, n_hidden),
                             nn.ReLU(),
                             nn.Linear(n_hidden, n_out),
                             nn.Sigmoid())

    loss_function = nn.MSELoss() # TODO: check which loss function alternative nn.BCELoss()
    optimizer = torch.optim.SGD(ml_model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(5000):
        pred_y = ml_model(data_x)
        loss = loss_function(pred_y, data_y)
        losses.append(loss.item())

        ml_model.zero_grad()
        loss.backward()

        optimizer.step()
        # print(f'Finished epoch {epoch}, latest loss {loss}')

    # import matplotlib.pyplot as plt
    # plt.plot(losses)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.title("Learning rate %f"%(learning_rate))
    # plt.show()

    coef = pd.DataFrame(0, index=X_train.columns.values, columns=['value']) # TODO: add value?

    ml_model.name = "nn"
    return ml_model, coef

