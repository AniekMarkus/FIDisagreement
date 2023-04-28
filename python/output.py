
# Modules
import argparse
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

from pathlib import Path
from datetime import date

os.environ["CUDA_VISIBLE_DEVICES"]=""

# Get functions in other Python scripts
from evaluation_metrics import *

import warnings
warnings.filterwarnings('ignore')

def aggregate(data, output_folder):
    fi_files = os.listdir(output_folder / "feature_importance")
    fi_files = list(filter(lambda v: re.findall(data, v), fi_files))

    feature_importance = pd.DataFrame(columns=['data', 'repeat', 'model', 'fi_method'])

    for fi in fi_files:  # fi = fi_files[0]
        fi_values = pd.read_csv(output_folder / "feature_importance" / fi, index_col=0)

        params = fi.split(sep="-")
        info = pd.Series({'data': params[0],
                          'repeat': params[1],
                          'model': params[2],
                          'fi_method': params[3]}, dtype='object')

        feature_importance = feature_importance.append(info.append(pd.Series(fi_values.value, index=fi_values.index)), ignore_index=True)
        # TODO: fix without ignore_index true

    feature_importance.transpose().to_csv(f'{output_folder}/result/{data}-aggregate.csv', header=False)

    return feature_importance

def evaluate_disagreement(data, feature_importance, output_folder):
    metrics = ['overlap', 'rank', 'sign', 'ranksign', 'pearson', 'kendalltau', 'pairwise_comp', 'mae', 'rmse', 'r2']
    # feature_importance = pd.read_csv(f'{output_folder}/result/{data}-aggregate.csv', index_col=0)

    results = pd.DataFrame(columns=['data1', 'repeat1', 'model1', 'fi_method1',
                                    'data2', 'repeat2', 'model2', 'fi_method2'])

    # Calculate evaluation metrics
    for i, row in feature_importance.iterrows():
        for j in range(i+1):
            cols = ['data', 'repeat', 'model', 'fi_method']
            fi_meth1 = feature_importance.iloc[i]
            fi_meth2 = feature_importance.iloc[j]

            eval_metrics = fi_evaluate(fi1=fi_meth1.drop(cols),
                                       fi2=fi_meth2.drop(cols),
                                       metrics=metrics)

            info = pd.Series({'data1': fi_meth1['data'],
                              'repeat1': fi_meth1['repeat'],
                              'model1': fi_meth1['model'],
                              'fi_method1': fi_meth1['fi_method'],
                              'data2': fi_meth2['data'],
                              'repeat2': fi_meth2['repeat'],
                              'model2': fi_meth2['model'],
                              'fi_method2': fi_meth2['fi_method']}, dtype='object')

            results = results.append(info.append(pd.Series(eval_metrics, index=metrics)), ignore_index=True)

    # Add names to columns
    results.to_csv(f'{output_folder}/result/{data}-eval_metrics.csv', index=False)

    return results


if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    i=None

    # SETTINGS
    root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"
    output_folder = Path(root_folder + "/results/output_" + str(date.today()))

    # PARAMETERS
    parser = argparse.ArgumentParser(
        prog = 'OutputFI',
        description = 'Aggregate output and evaluate resuls',
        epilog = '...')

    parser.add_argument('data')
    args = parser.parse_args()
    print(f'Parameters : data = {args.data}')

    # CREATE FOLDERS
    result_folder = output_folder / "result"
    if not result_folder.exists():
        os.mkdir(result_folder)

    # EVALUATE
    # aggregate results
    feature_importance = aggregate(args.data, output_folder)

    # compute disagreement
    eval_metrics = evaluate_disagreement(args.data, feature_importance, output_folder)

    # compute axioms
    # eval_metrics = evaluate_axioms(args.data, feature_importance, output_folder)



