from pathlib import Path
import pandas as pd
from datetime import date

root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"
output_folder = Path(root_folder + "/results/output_" + str(date.today()))

final_evaluation = pd.read_csv(output_folder / "final_evaluation.csv")
final_evaluation['fi_meth'] = final_evaluation.fi_meth1 + "_" + final_evaluation.fi_meth2

data_reference = pd.read_csv(output_folder / "data_reference.csv")
# data_reference['name'] = ["data" + str(i) for i in data_reference.index]

# color_dict = {'permutation': '#1f77b4',
#               'shap': '#ff7f0e',
#               'sage': '#2ca02c',
#               'kernelshap': '#d62728'}

color_dict = {'coefficient': '#ecda9a',
              'permutation_auc': '#f3ad6a',
              'permutation_rmse': '#f66356',
              'permutation_accuracy': '#f7945d',
              'permutation_balanced': '#f97b57',
              'kernelshap_500': '#96d2a4',
              'kernelshap_1000': '#4da284',
              #'kernelshap_3000': '#2a5674',
              #'sage_500': '#68abb8',
              #'sage_1000': '#2a5674',
              'marginalsage_1000': '#68abb8',
              'conditionalsage_1000': '#2a5674'}
# defaults ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
