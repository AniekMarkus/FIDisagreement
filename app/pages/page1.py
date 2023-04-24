import dash

dash.register_page(__name__)

from dash import dcc, html, dash_table,  Input, Output, callback
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

from app.results_explorer_input import output_folder, data_reference, final_evaluation, color_dict
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
                                            options=final_evaluation.name.unique(),
                                            clearable=False,
                                            searchable=False,
                                            value=final_evaluation.name.unique()[0],
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Model",
                                            id="dropdown-select-model",
                                            options=final_evaluation.model.unique(),
                                            clearable=False,
                                            searchable=False,
                                            value=final_evaluation.model.unique()[0],
                                        ),
                                        drc.NamedDropdown(
                                            name="Select FI Method",
                                            id="dropdown-select-fimethod",
                                            options=final_evaluation.fi_meth1.append(final_evaluation.fi_meth2).unique().tolist(),
                                            clearable=False,
                                            searchable=False,
                                            value=final_evaluation.fi_meth1.append(final_evaluation.fi_meth2).unique().tolist(), # ["permutation", "shap", "kernelshap", "sage"]
                                            multi=True,
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Metrics",
                                            id="dropdown-select-metrics",
                                            options=final_evaluation.columns[~final_evaluation.columns.isin(['name', 'model', 'fi_meth1', 'fi_meth2', 'fi_meth'])],
                                            clearable=False,
                                            searchable=False,
                                            value=final_evaluation.columns[~final_evaluation.columns.isin(['name', 'model', 'fi_meth1', 'fi_meth2', 'fi_meth'])],
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
                                id="temp",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                            style={'margin': '210px'}
                        ),
                     ]
                 )

@callback(
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
    fi_plot = figs.serve_fi_visualization(output_folder, color_dict, dataset, fimethod)
    fi_rank = figs.fi_ranking(output_folder, dataset, fimethod)
    fi_top = figs.fi_topfeatures(output_folder, color_dict, dataset, fimethod, k=10)
    fi_metrics = figs.serve_fi_metrics(color_dict, final_evaluation, dataset, model, fimethod, metrics)
    data_ref = pd.DataFrame(data_reference.loc[data_reference.name == dataset, :])

    return [
        html.Div(
            id="graph-container1",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="fi_plot", figure=fi_plot),
                style={'float': 'none'},
            ),
        ),
        html.Div(
            id="graph-container12",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="fi_rank", figure=fi_rank),
                    style={'float': 'none'},
                ),
            ],
        ),
        html.Div(
            id="graph-container13",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="fi_top", figure=fi_top),
                    style={'float': 'none'},
                ),
            ],
        ),
        html.Div(
            id="graph-container2",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="fi_metrics", figure=fi_metrics),
                    style={'float': 'none'},
                ),
            ],
        ),
        html.Div(
            id="graph-container3",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dash_table.DataTable(data_ref.to_dict('records'), [{"name": "data" + str(i), "id": i} for i in data_ref.columns]),
                    style={'float': 'none'},
                ),
            ],
        ),
    ]
