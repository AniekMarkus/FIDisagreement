# Modules
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns  # correlogram
import warnings

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import xgboost

import os
import pickle
import shutil

from sklearn import metrics

# from sklearn_gbmi import *  # friedman H statistic
from sklearn.ensemble import GradientBoostingRegressor  # friedman H statistic

# Get functions in other Python scripts
def setup(output_folder, clean):
    if clean and output_folder.exists():
        shutil.rmtree(output_folder)

    if not output_folder.exists():
        os.mkdir(output_folder)
        os.mkdir(output_folder / "data")
        os.mkdir(output_folder / "models")
        os.mkdir(output_folder / "feature_importance")
        os.mkdir(output_folder / "eval")

def split_data(X, y, test_size = 0.25):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=2022, test_size=test_size)

    return X_train, X_test, y_train, y_test

def normalise(array):
    if (array.max() - array.min()) != 0:
        return ((array - array.min()) / (array.max() - array.min()))
    else:
        warnings.warn("Array not normalised: array.max() - array.min()) = 0.")
        return array


def wrapper_predict(ml_model, X, prob=True):
    if ml_model.name == 'nn':
        with torch.no_grad():
            if not isinstance(X, np.ndarray):
                data_x = torch.tensor(X.values, dtype=torch.float32)
            else:
                data_x = torch.tensor(X, dtype=torch.float32)

            pred_prob = ml_model(data_x)
            pred_prob = pred_prob.detach().numpy()
    else:
        if ml_model.name == "xgboost":
            if not isinstance(X, np.ndarray):
                data_x = xgboost.DMatrix(X.values)
            else:
                data_x = xgboost.DMatrix(X)

            pred_prob = ml_model.predict(data_x)
        elif ml_model.name == "logistic":
            if not isinstance(X, np.ndarray):
                data_x = X.values
            else:
                data_x = X

            pred_prob = ml_model.predict_proba(data_x)
            pred_prob = pred_prob[:, 1]

    if prob:
        return pred_prob
    else:
        return np.where(pred_prob < 0.5, 0, 1)



def test_model(output_folder, ml_model, coef_model, X_test, y_test, model, data):
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



def correlation(x1, x2):
    E_x1 = np.mean(x1)
    Var_x1 = np.mean(x1**2) - E_x1**2
    sigma_x1 = np.sqrt(Var_x1)

    E_x2 = np.mean(x2)
    Var_x2 = np.mean(x2**2) - E_x2**2
    sigma_x2 = np.sqrt(Var_x2)

    Cov_x1_x2 = np.mean(x1*x2) - E_x1*E_x2
    Corr_x1_x2 = Cov_x1_x2/sigma_x1/sigma_x2

    print('corr(S1,S2) = ' + str(Corr_x1_x2))

    return Corr_x1_x2


def correlogram(X, pdf):
    # Compute correlations
    # corr = X.drop('intercept', axis=1).corr()
    corr = X.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Alternative: basic correlogram
    # sns.pairplot(X)

    plt.title('Correlogram of features')
    pdf.savefig()
    plt.show()


# def friedmanH(X, y, pdf):
#     seed = 2022
#
#     gbr_1 = GradientBoostingRegressor(random_state=seed)
#     # gbr_1.fit(X.drop('intercept', axis=1), y)
#     gbr_1.fit(X, y)
#
#     # res = h_all_pairs(gbr_1, X.drop('intercept', axis=1))
#     res = h_all_pairs(gbr_1, X)
#
#     res = pd.DataFrame.from_dict(res, orient="index")
#     res = res.reset_index()
#     res.columns = ['var_pair', 'H_stat']
#
#     names = res['var_pair'].astype('str').str.split(',', expand=True)
#     res['var1'] = names[0].str.replace('(', '', regex=False)
#     res['var2'] = names[1].str.replace(')', '', regex=False)
#
#     H_stat = res.pivot(index='var2', columns='var1', values='H_stat')
#
#     # Generate a custom diverging colormap
#     cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
#     # Draw the heatmap with the mask and correct aspect ratio
#     sns.heatmap(H_stat, cmap=cmap, vmax=.3, center=0,
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
#     plt.title('Friedman H statistic of features')
#     pdf.savefig()
#     plt.show()
#
#     # TODO: check if all variables included?
