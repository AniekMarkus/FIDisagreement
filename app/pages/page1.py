import dash
import pyodbc

dash.register_page(__name__)

from dash import dcc, html, dash_table,  Input, Output, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app.results_explorer_input import output_folder, datasets, versions, models, fimethods, eval_metrics, color_dict
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
                                            name="Select Version",
                                            id="dropdown-select-version",
                                            options=versions,
                                            clearable=False,
                                            searchable=False,
                                            value="v0",
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
                                            name="Select Metrics",
                                            id="dropdown-select-metrics",
                                            options=eval_metrics,
                                            clearable=False,
                                            searchable=False,
                                            value=eval_metrics,
                                            multi=True,
                                        ),
                                    ],
                                )
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
        Input("dropdown-select-version", "value"),
        Input("dropdown-select-model", "value"),
        Input("dropdown-select-fimethod", "value"),
        Input("dropdown-select-metrics", "value")
    ],
)
def update_graph_top(
        dataset,
        version,
        model,
        fimethod,
        eval_metrics
):
    fi_plot = figs.fi_values(output_folder, color_dict, dataset, version, model, fimethod)
    fi_rank = figs.fi_ranking(output_folder, color_dict, dataset, version, model, fimethod)
    fi_top = figs.fi_topfeatures(output_folder, color_dict, dataset, version, model, fimethod, k=5)
    fi_metrics = figs.fi_metrics(output_folder, color_dict, dataset, version, model, fimethod, eval_metrics)

    return [
        html.Div([
        dbc.Row([
        dbc.Col(html.Div(
            id="graph-container1",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="fi_plot", figure=fi_plot),
                style={'float': 'none'},
            ),
        )),
        dbc.Col(html.Div(
            id="graph-container12",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="fi_rank", figure=fi_rank),
                    style={'float': 'none'},
                ),
            ],
        ))])]),
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
        )
    ]
