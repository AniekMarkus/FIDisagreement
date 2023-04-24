
import os
import re
import pandas as pd


def get_fi(output_folder,  dataset="data2"):
    all_feature_importance = os.listdir(output_folder / "feature_importance")

    feature_importance = pd.DataFrame()
    for f in list(filter(lambda v: re.findall(dataset, v), all_feature_importance)):
        file = pd.read_csv(output_folder / "feature_importance" / f) # index_col=0
        feature_importance = feature_importance.append(file, ignore_index=True)
    feature_importance = feature_importance.rename(columns={feature_importance.columns[0]: 'variable'})  # add column name

    return feature_importance