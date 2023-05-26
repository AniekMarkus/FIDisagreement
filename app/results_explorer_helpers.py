
import os
import re
import pandas as pd

from app.results_explorer_input import *
from python.help_functions import *

def get_fi(output_folder, dataset, version, model, fimethod, scale=False):
    # Load file
    feature_importance = pd.read_csv(output_folder / "result" / str(dataset + "-" + model + "-aggregate.csv"), header=None, index_col=0)

    # Correct input type
    if not isinstance(version, list):
        version = [version]
    if not isinstance(model, list):
        model = [model]
    if not isinstance(fimethod, list):
        fimethod = [fimethod]

    # Filter results
    # filter = [col for col in feature_importance.columns if feature_importance.loc["fi_method", col] in fimethod]
    filter0 = feature_importance.loc["version"].isin(version)
    filter1 = feature_importance.loc["model"].isin(model)
    filter2 = feature_importance.loc["fi_method"].isin(fimethod)
    feature_importance = feature_importance.loc[:, filter0 & filter1 & filter2]

    # Add column names
    feature_importance.columns = ['settings_' + str(col) for col in feature_importance.columns]
    feature_importance.index.name = 'variable'

    # Split results
    fi_values = feature_importance.drop(settings_cols, axis=0)
    fi_values = fi_values.apply(lambda c: c.astype(float), axis=0)

    # Scale values
    if scale:
        fi_values = fi_values.apply(lambda c: normalise(c), axis=0)

    settings_data = feature_importance.loc[settings_cols, :]

    return fi_values, settings_data


def get_rank(output_folder, dataset, version, model, fimethod, scale=False):

    # Get feature importance values
    fi_values, settings_data = get_fi(output_folder, dataset, version, model, fimethod, scale)

    # Take absolute value
    abs_values = fi_values.apply(lambda c: c.abs(), axis=0)

    # Transform to rank
    fi_rank = abs_values.apply(lambda c: c.rank(ascending=False, method='dense'), axis=0)

    # TODO: save fi_rank.to_csv(output_folder / 'ranking_abs_scaled.csv')

    return fi_rank, fi_values, settings_data


def combine_fi(fi, settings_data):
    # Add variable names
    fi.reset_index(inplace=True, drop=False)

    # Translate wide to long format
    fi = pd.melt(fi, id_vars= "variable", var_name="method")

    # Combine plot_data with settings_data
    settings_data = settings_data.transpose()
    settings_data.index.name = 'method'
    settings_data.reset_index(drop=False, inplace=True)

    fi = pd.merge(left=fi, right= settings_data, left_on = 'method', right_on='method')
    fi.drop(["method", "version", "model", "data"], inplace=True, axis=1)

    return fi


def get_metrics(output_folder, dataset, version, model, fimethod, eval_metrics, summarize=False):
    # Load file
    res_metrics = pd.read_csv(output_folder / "result" / str(f"{dataset}-{version}-{model}-eval_metrics.csv"))
    res_metrics = res_metrics.loc[(res_metrics.fi_method1.isin(fimethod) & res_metrics.fi_method2.isin(fimethod)), :]

    # Save names
    eval_names = res_metrics.loc[:, res_metrics.columns.isin(['repeat', 'fi_method1', 'fi_method2'])]

    # Correct input type
    if not isinstance(eval_metrics, list):
        eval_metrics = [eval_metrics]

    # Select metrics
    res_metrics = res_metrics.loc[:, res_metrics.columns.isin(eval_metrics)]

    # All metrics to same scale
    # Between 0 and 1 (higher = agreement): 'overlap', 'rank', 'sign', 'ranksign', 'pairwise_comp'(???)

    # Between -1 and 1 (higher = agreement): 'kendalltau', 'pearson'
    # Normalise values
    cols = res_metrics.columns.isin(['kendalltau', 'pearson'])
    res_metrics.loc[:, cols]=res_metrics.loc[:, cols].apply(lambda c: (c - (-1)) / (1 - (-1)), axis=0)

    # Minimum 0 (lower = agreeement) : 'mae', 'rmse'
    # Normalise + reverse values (so higher is always more agreement)
    cols = res_metrics.columns.isin(['mae', 'rmse'])
    res_metrics.loc[:, cols]=res_metrics.loc[:, cols].apply(lambda c: 1-normalise(c), axis=0)

    # Positive and negative values (higher = agreement): 'r2'
    # Normalise values
    cols = res_metrics.columns.isin(['r2'])
    res_metrics.loc[:, cols]=res_metrics.loc[:, cols].apply(lambda c: normalise(c), axis=0)

    # Take mean across different metrics
    if summarize:
        res_metrics = res_metrics.mean(axis=1)

    return res_metrics, eval_names
