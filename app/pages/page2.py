import dash

dash.register_page(__name__)

from dash import dcc, html, dash_table,  Input, Output, callback
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from app.results_explorer_input import output_folder, datasets, models, fimethods, eval_metrics, color_dict, modify_params
import app.results_explorer_utils as drc
import app.results_explorer_figures as figs

layout = html.Div(id="app-container",  # id="app-container",
                  # className="container scalable", # className="row",
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
                                          options=datasets,
                                          clearable=False,
                                          searchable=False,
                                          value=datasets[0],
                                      ),
                                      drc.NamedDropdown(
                                          name="Select Model",
                                          id="dropdown-select-model",
                                          options=models,
                                          clearable=False,
                                          searchable=False,
                                          value=models[0],
                                      ),
                                      drc.NamedDropdown(
                                          name="Select FI Method",
                                          id="dropdown-select-fimethod",
                                          options=fimethods,
                                          clearable=False,
                                          searchable=False,
                                          value=fimethods,
                                          multi=True,
                                      ),
                                      drc.NamedDropdown(
                                          name="Select Metric (y-axis)",
                                          id="dropdown-select-metric",
                                          options=eval_metrics,
                                          clearable=False,
                                          searchable=False,
                                          value="mae",
                                          multi=False,
                                      )
                          ],
                              )
                          ],
                      ),
                      html.Div(
                          id="div-graphs2",
                          children=dcc.Graph(
                              id="complexity_plot",
                              figure=dict(
                                  layout=dict(
                                      plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                  )
                              ),
                          ),
                      )
                  ]
                  )


@callback(
    Output("div-graphs2", "children"),
    [
        Input("dropdown-select-dataset", "value"),
        Input("dropdown-select-model", "value"),
        Input("dropdown-select-fimethod", "value"),
        Input("dropdown-select-metric", "value")
    ],
)
def update_graph_bottom(
    dataset,
    model,
    fimethod,
    metric
):
    complexity_features = figs.complexity_plot(output_folder, color_dict, modify_params, dataset, "v1", model, fimethod, metric)
    complexity_observations = figs.complexity_plot(output_folder, color_dict, modify_params, dataset, "v2", model, fimethod, metric)
    complexity_outcomes = figs.complexity_plot(output_folder, color_dict, modify_params, dataset, "v3", model, fimethod, metric)
    complexity_correlation = figs.complexity_plot(output_folder, color_dict, modify_params, dataset, "v4", model, fimethod, metric)
    complexity_prevalance = figs.complexity_plot(output_folder, color_dict, modify_params, dataset, "v5", model, fimethod, metric)

    return [
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_features),
                style={"display": "none"}
            )
        ),
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_observations),
                style={"display": "none"}
            )
        ),
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_outcomes),
                style={"display": "none"}
            )
        ),
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_correlation),
                style={"display": "none"}
            )
        ),
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_prevalance),
                style={"display": "none"}
            )
        )
    ]