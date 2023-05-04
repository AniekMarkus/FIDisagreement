from pathlib import Path
import pandas as pd
from datetime import date
import os
import re

root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"
output_folder = Path(root_folder + "/results/output_" + str(date.today()))

settings_cols = ['data', 'repeat', 'version', 'model', 'fi_method']

files = os.listdir(output_folder / "feature_importance")
datasets = list(set(map(lambda v: v.split(sep="-")[0], files)))
versions = list(set(map(lambda v: v.split(sep="-")[2], files)))
models = list(set(map(lambda v: v.split(sep="-")[3], files)))
fimethods = list(set(map(lambda v: v.split(sep="-")[4], files)))

metrics = ['overlap', 'rank', 'sign', 'ranksign',
           'pearson', 'kendalltau', 'pairwise_comp', 'mae', 'rmse', 'r2']

color_dict = {'coefficient': '#ecda9a',
              'permutation_auc': '#f3ad6a',
              'permutation_mse': '#f66356',
              'permutation_accuracy': '#f7945d',
              'permutation_ba': '#f97b57',
              'kernelshap': '#96d2a4',
              #'kernelshap_1000': '#4da284',
              'sage_marginal': '#68abb8',
              'sage_conditional': '#2a5674'}
# defaults ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fileName = output_folder / "data" / "modify_params.csv"

if os.path.exists(fileName):
    modify_params = pd.read_csv(fileName, header=0)
    modify_params.rename(columns={'Unnamed: 0': 'version'}, inplace=True)
    modify_params.version= ['v' + str(row) for row in modify_params.version]
else:
    modify_params = {'version': 'v0'}