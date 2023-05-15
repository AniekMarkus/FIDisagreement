
# Modules
import numpy as np
import pandas as pd
import os

import argparse

import re

from pathlib import Path
from datetime import date

# Get functions in other Python scripts
from evaluation_metrics import *

import warnings
warnings.filterwarnings('ignore')

def aggregate(data, model, output_folder):
    fi_files = os.listdir(output_folder / "feature_importance")
    fi_files = list(filter(lambda v: re.findall(data, v), fi_files))
    fi_files = list(filter(lambda v: re.findall(model, v), fi_files))

    feature_importance = pd.DataFrame(columns=['data', 'repeat', 'version', 'model', 'fi_method'])

    for fi in fi_files:  # fi = fi_files[0]
        fi_values = pd.read_csv(output_folder / "feature_importance" / fi, index_col=0)

        params = fi.split(sep="-")
        info = pd.Series({'data': params[0],
                          'repeat': params[1],
                          'version': params[2],
                          'model': params[3],
                          'fi_method': params[4]}, dtype='object')

        feature_importance = feature_importance.append(info.append(pd.Series(fi_values.value, index=fi_values.index)), ignore_index=True)
        # TODO: fix without ignore_index true

    feature_importance.transpose().to_csv(f'{output_folder}/result/{data}-{model}-aggregate.csv', header=False)

    print("> aggregate done")
    return feature_importance

def evaluate_disagreement(data, feature_importance, output_folder):
    eval_metrics = ['overlap', 'rank', 'sign', 'ranksign', 'pearson', 'kendalltau', 'pairwise_comp', 'mae', 'rmse', 'r2']
    # feature_importance = pd.read_csv(f'{output_folder}/result/{data}-aggregate.csv', index_col=0)

    results = pd.DataFrame(columns=['data', 'model', 'fi_method1', 'fi_method2'])

    # Impute missings with zero (indicating no model importance)
    feature_importance = feature_importance.fillna(0)  # TODO: or remove these var?

    # Average values over repeats
    feature_importance = feature_importance.groupby(by=['data', 'version', 'model', 'fi_method'], group_keys=True, as_index=False).mean()

    # Calculate evaluation metrics
    for i, row in feature_importance.iterrows():
        for j, row in feature_importance.iterrows():
        # for j in range(i+1):
            cols = ['data', 'version', 'model', 'fi_method']
            fi_meth1 = feature_importance.iloc[i]
            fi_meth2 = feature_importance.iloc[j]

            res_metrics = fi_evaluate(fi1=fi_meth1.drop(cols),
                                       fi2=fi_meth2.drop(cols),
                                       eval_metrics=eval_metrics)

            info = pd.Series({'data': fi_meth1['data'],
                              'model': fi_meth1['model'],
                              'fi_method1': fi_meth1['fi_method'],
                              'fi_method2': fi_meth2['fi_method']}, dtype='object')

            results = results.append(info.append(pd.Series(res_metrics, index=eval_metrics)), ignore_index=True)

    # Add names to columns
    results.to_csv(f'{output_folder}/result/{data}-eval_metrics.csv', index=False)

    print("> evaluate_disagreement done")
    return results


if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    i=None

    # SETTINGS
    root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"

    # PARAMETERS
    parser = argparse.ArgumentParser(
        prog = 'OutputFI',
        description = 'Aggregate output and evaluate resuls',
        epilog = '...')

    parser.add_argument('data')
    parser.add_argument('model')
    parser.add_argument('folder', default=None)
    args = parser.parse_args()
    print(f'Parameters : data = {args.data}, model = {args.model}')

    # If no folder given, use output_folder
    if args.folder is None:
        output_folder = Path(root_folder + "/results/output_" + str(date.today()))
    else:
        output_folder = Path(root_folder + "/results/" + str(args.folder))

    # CREATE FOLDERS
    result_folder = output_folder / "result"
    if not result_folder.exists():
        os.mkdir(result_folder)
        os.mkdir(output_folder / "plots")

    # EVALUATE
    # aggregate results
    feature_importance = aggregate(args.data, args.model, output_folder)

    # compute disagreement for each version of data and model
    for v in feature_importance.version.unique():
        feature_importance_p = feature_importance.loc[feature_importance.version == v, :]
        res_metrics_p = evaluate_disagreement(str(f"{args.data}-{v}-{args.model}"), feature_importance_p, output_folder)

    # compute axioms
    # eval_metrics = evaluate_axioms(args.data, feature_importance, output_folder)

