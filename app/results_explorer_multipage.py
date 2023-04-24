import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import os

assets_path = os.getcwd() + '/app/assets'

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP],
                pages_folder="/app/pages", assets_folder=assets_path)

app.title = "Feature Importance Explorer"

navbar = html.Div(
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
                html.Img(src=app.get_asset_url("logo.png"), width="40px", height="30px")
            ],
            href="https://plot.ly/products/dash/",
        ),
        html.I(id="dropdown",
               children=[
                   dbc.DropdownMenu(
                       [
                           dbc.DropdownMenuItem(page["name"], href=page["path"])
                           for page in dash.page_registry.values()
                           if page["module"] != "pages.not_found_404"
                       ],
                       nav=True,
                       label="Pages",
                   ),
               ],
               )
    ],
)

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[navbar],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[dash.page_container],
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
