from pathlib import Path
import pandas as pd
from datetime import date
import os
import re
import plotly.graph_objs as go

root_folder = "..."
output_folder = Path(root_folder + "/results/output_2023-ICML")

settings_cols = ['data', 'repeat', 'version', 'model', 'fi_method']

files = os.listdir(output_folder / "feature_importance")

color_dict = {'coefficient': '#ecda9a',
              'permutation_auc': '#f3ad6a',
              'permutation_mse': '#f66356',
              'permutation_ba': '#f97b57',
              'loco_auc': '#9467bd',
              'loco_mse': '#ba9cd4',
              'loco_ba': '#dfd1eb',
              'kernelshap': '#96d2a4',
              'sage_marginal': '#68abb8',
              'sage_conditional': '#2a5674'}

FI_name_dict = {'coefficient': 'coefficient',
              'permutation_auc': 'PFI AUC',
              'permutation_mse': 'PFI MSE',
              'permutation_ba': 'PFI BA',
              'loco_auc': 'LOCO AUC',
              'loco_mse': 'LOCO MSE',
              'loco_ba': 'LOCO BA',
              'kernelshap': 'KernelSHAP',
              'sage_marginal': 'SAGE-M',
              'sage_conditional': 'SAGE-C'}

fileName = output_folder / "data" / "modify_params.csv"

if os.path.exists(fileName):
    modify_params = pd.read_csv(fileName, header=0)
    modify_params.rename(columns={'Unnamed: 0': 'version'}, inplace=True)
    modify_params.version= ['v' + str(row) for row in modify_params.version]
else:
    modify_params = {'version': 'v0'}

# PAPER
std_layout = go.Layout(
    margin=dict(l=100, r=10, t=50, b=40),
    showlegend=False
)

# APP
# std_layout = go.Layout(
#     xaxis=dict(gridcolor="#2f3445"),
#     yaxis=dict(gridcolor="#2f3445"),
#     # legend=dict(x=0, y=1.05, orientation="v"),
#     margin=dict(l=100, r=10, t=50, b=40),
#     plot_bgcolor="#282b38",
#     paper_bgcolor="#282b38",
#     font={"color": "#a5b1cd"},
# )
