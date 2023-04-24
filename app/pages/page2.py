import dash

dash.register_page(__name__)

from dash import dcc, html, dash_table,  Input, Output, callback
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from app.results_explorer_input import output_folder, data_reference, final_evaluation
import app.results_explorer_utils as drc
import app.results_explorer_figures as figs

combined = pd.merge(data_reference, final_evaluation, on="name")

list_characteristics = data_reference.columns[~data_reference.columns.isin(['data'])]
list_metrics = final_evaluation.columns[~final_evaluation.columns.isin(['data', 'model', 'fi_meth1', 'fi_meth2', 'fi_meth'])]

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
                                              # drc.NamedDropdown(
                                              #     name="Select Characteristic (x-axis)",
                                              #     id="dropdown-select-characteristic",
                                              #     options=list_characteristics,
                                              #     clearable=False,
                                              #     searchable=False,
                                              #     value="F",
                                              #     multi=False,
                                              # ),
                                              drc.NamedDropdown(
                                                  name="Select Metric (y-axis)",
                                                  id="dropdown-select-metric",
                                                  options=list_metrics,
                                                  clearable=False,
                                                  searchable=False,
                                                  value="mae",
                                                  multi=False,
                                              )
                                  ],
                              ),
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


# layout = html.Div(
#     [
#         dcc.Graph(id="complexity_plot"),
#         drc.NamedDropdown(
#             name="Select Characteristic (x-axis)",
#             id="dropdown-select-characteristic",
#             options=list_characteristics,
#             clearable=False,
#             searchable=False,
#             value="F",
#             multi=False,
#         ),
#         drc.NamedDropdown(
#             name="Select Metric (y-axis)",
#             id="dropdown-select-metric",
#             options=list_metrics,
#             clearable=False,
#             searchable=False,
#             value="mae",
#             multi=False,
#         ),
#         # drc.NamedRangeSlider(
#         #     name="Sample Size",
#         #     min=0, max=5000, step=1000, value=[0, 2000], id='sample-size-slider'
#         # ),
#         # drc.NamedRangeSlider(
#         #     name="Informative Features (%)",
#         #     min=0, max=100, step=10, value=[20, 50], id='informative-features-slider'
#         # ),
#     ]
# )


@callback(
    Output("div-graphs2", "children"),
    [
       # Input("dropdown-select-characteristic", "value"),
        Input("dropdown-select-metric", "value")
    ],
)
def update_graph_bottom(
        # characteristic,
        metric
):
    # fi_plot = figs.serve_fi_visualization(output_folder, dataset="data1", model="model-logistic", fimethod="shap")
    complexity_plot_rho = figs.serve_complexity_plot(output_folder, combined, list_characteristics, list_metrics, "rho", metric)
    complexity_plot_F = figs.serve_complexity_plot(output_folder, combined, list_characteristics, list_metrics, "F", metric)
    complexity_plot_inf = figs.serve_complexity_plot(output_folder, combined, list_characteristics, list_metrics, "inf", metric)

    return [
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_plot_rho),
                style={"display": "none"}
            )
        ),
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_plot_F),
                style={"display": "none"}
            )
        ),
        html.Div(
            id="graph-container0",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="complexity_plot", figure=complexity_plot_inf),
                style={"display": "none"}
            )
        )
    ]
