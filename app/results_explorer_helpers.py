
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
    fi_rank = abs_values.apply(lambda c: c.rank(ascending=False, method='min'), axis=0)

    # TODO: save fi_rank.to_csv(output_folder / 'ranking_abs_scaled.csv')

    return fi_rank, fi_values, settings_data


def combine_fi(fi, settings_data):
    # Add variable names
    fi.reset_index(inplace=True, drop=False)

    # Save order of variables
    # order_variable = fi_values.variable

    # Translate wide to long format
    fi = pd.melt(fi, id_vars= "variable", var_name="method")

    # Combine plot_data with settings_data
    settings_data = settings_data.transpose()
    settings_data.index.name = 'method'
    settings_data.reset_index(drop=False, inplace=True)

    fi = pd.merge(left=fi, right= settings_data, left_on = 'method', right_on='method')
    fi.drop(["method", "version", "model", "data"], inplace=True, axis=1)

    return fi


def get_metrics(output_folder, dataset, version, model, fimethod, metrics):
    # Load file
    eval_metrics = pd.read_csv(output_folder / "result" / str(f"{dataset}-{version}-{model}-eval_metrics.csv"))
    eval_metrics = eval_metrics.loc[(eval_metrics.fi_method1.isin(fimethod) | eval_metrics.fi_method2.isin(fimethod)), :]

    # Save names
    eval_names = eval_metrics.loc[:, eval_metrics.columns.isin(['fi_method1', 'fi_method2'])]

    # Correct input type
    if not isinstance(metrics, list):
        metrics = [metrics]

    eval_metrics = eval_metrics.loc[:, eval_metrics.columns.isin(metrics)]

    # Reverse values (so higher is always more agreement)
    cols = eval_metrics.columns.isin(['mae', 'rmse', 'r2'])
    eval_metrics.loc[:, cols]=eval_metrics.loc[:, cols].apply(lambda c: 1-normalise(c), axis=0)

    return eval_metrics, eval_names
