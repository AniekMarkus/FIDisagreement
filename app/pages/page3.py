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
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphsb",
                            children=dcc.Graph(
                                id="tempb",
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
    Output("div-graphsb", "children"),
    [
        Input("dropdown-select-dataset", "value"),
    ],
)
def update_graph_p3(
        dataset
):
    fi_correlation = figs.data_correlogram(output_folder, dataset)
    data_ref = pd.DataFrame(data_reference.loc[data_reference.name == dataset, :])

    return [
        html.Div(
            id="graph-container1b",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="fi_correlation", figure=fi_correlation),
                style={'float': 'none'},
            ),
        ),
        html.Div(
            id="graph-container3b",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dash_table.DataTable(data_ref.to_dict('records'), [{"name": "data" + str(i), "id": i} for i in data_ref.columns]),
                    style={'float': 'none'},
                ),
            ],
        ),
    ]
