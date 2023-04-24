import time
import importlib

import dash
import pandas as pd
from dash import dcc, html, dash_table
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from pathlib import Path

# from results_explorer_utils import *
# from results_explorer_figures import *

import results_explorer_utils as drc
import results_explorer_figures as figs

from evaluation_metrics import *

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Feature Importance Explorer"
server = app.server

root_folder = "/Users/aniekmarkus/Documents/Git/FeatureImportancePython"
output_folder = Path(root_folder + "/output_2022-09-14")
# output_folder = Path(root_folder + "/output_" + str(date.today()))
final_evaluation = pd.read_csv(output_folder / "final_evaluation.csv")
data_reference = pd.read_csv(output_folder / "data_reference.csv")

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Feature Importance Explorer",
                                    href="https://github.com/aniekmarkus/featureimportance",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash-logo-new.png"))
                            ],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=final_evaluation.name.unique(),
                                            clearable=False,
                                            searchable=False,
                                            value="data1",
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Model",
                                            id="dropdown-select-model",
                                            options=final_evaluation.model.unique(),
                                            clearable=False,
                                            searchable=False,
                                            value="model-logistic",
                                        ),
                                        drc.NamedDropdown(
                                            name="Select FI Method",
                                            id="dropdown-select-fimethod",
                                            options=final_evaluation.fi_meth1.append(final_evaluation.fi_meth2).unique(),
                                            clearable=False,
                                            searchable=False,
                                            value=final_evaluation.fi_meth1.append(final_evaluation.fi_meth2).unique(), # ["permutation", "shap", "kernelshap", "sage"]
                                            multi=True,
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Metrics",
                                            id="dropdown-select-metrics",
                                            options=final_evaluation.columns[~final_evaluation.columns.isin(['data', 'model', 'fi_meth1', 'fi_meth2'])],
                                            clearable=False,
                                            searchable=False,
                                            value=final_evaluation.columns[~final_evaluation.columns.isin(['data', 'model', 'fi_meth1', 'fi_meth2'])],
                                            multi=True,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="second-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=0,
                                            max=5000,
                                            step=1000,
                                            marks={
                                                str(i): str(i)
                                                for i in [1000, 2000, 3000, 4000, 5000]
                                            },
                                            value=300,
                                        ),
                                        drc.NamedSlider(
                                            name="Informative Features",
                                            id="slider-dataset-informative-features",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="fi_plot",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)

@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-select-dataset", "value"),
        Input("dropdown-select-model", "value"),
        Input("dropdown-select-fimethod", "value"),
        Input("dropdown-select-metrics", "value")
    ],
)
def update_graph_top(
        dataset,
        model,
        fimethod,
        metrics
):
    fi_plot = figs.serve_fi_visualization(output_folder, dataset, model, fimethod)
    fi_metrics = figs.serve_fi_metrics(final_evaluation, dataset, model, fimethod, metrics)
    data_ref = pd.DataFrame(data_reference.iloc[int(dataset.replace("data", "")), :])

    return [
        html.Div(
            id="graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="fi_plot", figure=fi_plot),
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graph-container2",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="fi_metrics", figure=fi_metrics),
                ),
            ],
        ),
        html.Div(
            id="graph-container3",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dash_table.DataTable(data_ref.to_dict('records'), [{"name": "data" + str(i), "id": i} for i in data_ref.columns]),
                ),
            ],
        ),
    ]

# @app.callback(
#     Output("div-graphs", "children"),
#     [
#         Input("slider-dataset-sample-size", "value"),
#         Input("slider-dataset-informative-features", "value"),
#     ],
# )
# def update_graph_bottom(sample-size, informative_features):
#
#

# Running the server
if __name__ == "__main__":
    app.run_server(debug=False)

# import os
# import signal
# os.kill(os.getpid(), signal.SIGTERM)