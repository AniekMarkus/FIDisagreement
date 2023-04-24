
# Modules
import joblib
from pathlib import Path
from datetime import date
import os
import re
import pandas as pd
from scipy.io import arff
from sklearn import metrics
from sklearn.model_selection import KFold

# Python scripts
from python.help_functions import *
from python.simulate_data import *
from python.models import *
from python.feature_importance import *
from python.evaluation_metrics import *

# 1) Settings
# Output
root_folder = "/Users/aniekmarkus/Documents/Git/_Projects/FIDisagreement"
output_folder = Path(root_folder + "/results/output_" + str(date.today()))
save_folder = output_folder
# save_folder = Path(root_folder + "/results/output_2023-04-21")

if not output_folder.exists():
    os.mkdir(output_folder)
    os.mkdir(output_folder / "data")
    os.mkdir(output_folder / "models")
    os.mkdir(output_folder / "feature_importance")

if not save_folder.exists():
    os.mkdir(save_folder)
    os.mkdir(save_folder / "data")
    os.mkdir(save_folder / "models")
    os.mkdir(save_folder / "feature_importance")


# Input
repeats = 1
folds = 1
# use_data = "simulate"
use_data = "load"

# 2a) Simulate data
if use_data == "simulate":
    # Input
    data_reference = pd.read_csv('/input/input_settings.csv', index_col=0, sep=';')
    data_reference['name'] = 'data' + data_reference.index.astype(str)
    data_reference.to_csv(output_folder / "data_reference.csv")

    # Generate data using input parameters
    for i in range(1, repeats + 1):
        for d in range(1, data_reference.shape[0] + 1):
            [X, y] = simulate(N=data_reference.loc[d, 'N'],
                              o=data_reference.loc[d, 'o'],
                              beta=[int(i) for i in data_reference.loc[d, 'beta'].split(sep=",") if i.isdigit()],
                              F=data_reference.loc[d, 'F'],
                              inf=data_reference.loc[d, 'inf'],
                              t=data_reference.loc[d, 't'],
                              rho=data_reference.loc[d, 'rho'],
                              e=data_reference.loc[d, 'e'],
                              A=data_reference.loc[d, 'A'],
                              L=data_reference.loc[d, 'L'],
                              seed=2021+i)
            # Split data
            X_train, X_test, y_train, y_test = split_data(X, y) # Use train data, test data only for testing

            # Save data
            X_train.to_csv(output_folder / "data" / str("data" + str(d) + "_" + str(i) + "_Xtrain.csv"), index=False)
            pd.DataFrame(y_train).to_csv(output_folder / "data" / str("data" + str(d) + "_" + str(i) + "_ytrain.csv"), index=False)

            X_test.to_csv(output_folder / "data" / str("data" + str(d) + "_" + str(i) + "_Xtest.csv"), index=False)
            pd.DataFrame(y_test).to_csv(output_folder / "data" / str("data" + str(d) + "_" + str(i) + "_ytest.csv"), index=False)

# 2b) Load data
if use_data == "load":
    load_data = os.listdir(root_folder + "/input-data")
    load_data = ['iris.arff'] # TODO: change this?

    data_reference = pd.DataFrame()

    d = 1
    for data in load_data:  # data=load_data[0]
        # Import data
        input_data = arff.loadarff(root_folder + "/input-data/" + data)
        input_data = pd.DataFrame(input_data[0])

        # Split data
        X = input_data[input_data.columns[~input_data.columns.isin(['class'])]]
        y = input_data['class']

        # Scale data
        X = X.apply(lambda c: normalise(c), axis=0)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, 0.25) # Use train data, test data only for testing

        # Save data
        data_string = str(1) + "_" + re.sub(pattern=".arff", repl="", string=data)

        X_train.to_csv(output_folder / "data" / str("data" + str(d) + "_" + data_string + "_Xtrain.csv"), index=False)
        pd.DataFrame(y_train).to_csv(output_folder / "data" / str("data" + str(d) + "_" + data_string + "_ytrain.csv"), index=False)

        X_test.to_csv(output_folder / "data" / str("data" + str(d) + "_" + data_string + "_Xtest.csv"), index=False)
        pd.DataFrame(y_test).to_csv(output_folder / "data" / str("data" + str(d) + "_" + data_string + "_ytest.csv"), index=False)

        data_reference = data_reference.append(pd.Series({'name': "data" + str(d), 'full_name': data_string}, dtype='object'),
                                               ignore_index=True)
        d=d+1

    # Input
    data_reference.to_csv(output_folder / "data_reference.csv", index=False)


# 3) Train models
all_data = os.listdir(output_folder / "data")
all_data = list(filter(lambda v: re.findall("_Xtrain.csv", v), all_data))
all_data = list(map(lambda v: re.sub(pattern="_Xtrain.csv", repl="", string=v), all_data))

performance = pd.DataFrame()

for data in all_data:  # data = all_data[0]
    X_train = pd.read_csv(output_folder / "data" / str(str(data) + "_Xtrain.csv"))
    y_train = pd.read_csv(output_folder / "data" / str(str(data) + "_ytrain.csv"))
    # A column-vector y was passed when a 1d array was expected.

    d = re.search("data\d+", data).group(0)

    # TODO: save performance model? / (need train test split first?)
    # TODO: add random forest / neural network?
    if use_data == "load":
        model, coef_model = fit_logistic(X_train, y_train.iloc[:, 0]) # convert single column dataframe (row,1) to series (row,) format -> y = y[0].values
        joblib.dump(model, output_folder / "models" / str(str(data) + "_model-logistic.pkl"))
        np.savetxt(output_folder / "models" / str(str(data) + ".csv"), coef_model, delimiter=',', header=str(X_train.columns))

    elif data_reference.loc[data_reference.name == d, 'o'].values == 'Binary':  # TODO: combine this with statement above
        model, coef_model = fit_logistic(X_train, y_train.iloc[:, 0])
        joblib.dump(model, output_folder / "models" / str(str(data) + "_model-logistic.pkl"))
        np.savetxt(output_folder / "models" / str(str(data) + ".csv"), coef_model, delimiter=',', header=str(X_train.columns))

    elif data_reference.loc[data_reference.name == d, 'o'].values == 'Continuous':
        model, coef_model = fit_linear(X_train, y_train)
        joblib.dump(model, output_folder / "models" / str(str(data) + "_model-linear.pkl"))
        np.savetxt(output_folder / "models" / str(str(data) + ".csv"), coef_model, delimiter=',', header=str(X_train.columns))

    X_test = pd.read_csv(output_folder / "data" / str(str(data) + "_Xtest.csv"))
    y_test = pd.read_csv(output_folder / "data" / str(str(data) + "_ytest.csv"))

    pred = model.predict(X_test.values)
    pred_outcomes = pred.sum()
    # print('non-zeros:', pred_outcomes)

    pred = model.predict_proba(X_test.values)
    auc_score = metrics.roc_auc_score(y_test, pred[:, 1])
    auprc_score = metrics.average_precision_score(y_test, pred[:, 1])

    performance = performance.append(pd.Series({'name': re.search("data\d+", data).group(0),
                                                'non-zeros': pred_outcomes,
                                                'size': (coef_model != 0).sum(),
                                                'auc': auc_score,
                                                'auprc': auprc_score}, dtype='object'), ignore_index=True)

performance.to_csv(save_folder / "performance_model.csv", index=False)

# 4) Compute feature importance
all_models = os.listdir(output_folder / "models")
all_models = list(filter(lambda v: re.findall(".pkl", v), all_models))

times = pd.DataFrame()
# all_feature_importance = 0

# for i in range(1, repeats + 1):  # i = 1
for m in all_models:  # m = all_models[0]
    print(m)
    file_name = output_folder / "models" / m
    model = joblib.load(file_name)

    # TODO: adjust for load data file name
    # data = re.search("data\d+_\d+", m).group(0)
    data = re.sub(pattern="_model-[a-z]+.pkl", repl="", string=str(m))
    X_train = pd.read_csv(output_folder / "data" / str(data + "_Xtrain.csv"))
    y_train = pd.read_csv(output_folder / "data" / str(data + "_ytrain.csv"))

    coef_model = np.loadtxt(output_folder / "models" / str(str(data) + ".csv"), delimiter=',')

    # cross validation to compute robust feature importance
    # kf = KFold(n_splits=folds)
    # all_coefficients = 0

    i = 1
    # for incl_index, excl_index in kf.split(X_train):
    print('Computing FI for fold: ' + str(i))
    # X_subset, X_not = X_train.iloc[incl_index], X_train.iloc[excl_index]
    # y_subset, y_not = y_train.iloc[incl_index], y_train.iloc[excl_index]
    X_subset = X_train
    y_subset = y_train

    [permutation_auc_values, permutation_auc_time] = compute_permutation_custom(model, X_subset, y_subset, scoring='roc_auc')
    [permutation_rmse_values, permutation_rmse_time] = compute_permutation_custom(model, X_subset, y_subset,
                                                                                  scoring='neg_mean_squared_error')
    [permutation_balanced_values, permutation_balanced_time] = compute_permutation_custom(model, X_subset, y_subset,
                                                                                          scoring='balanced_accuracy')
    [permutation_auc_standard_values, permutation_auc_standard_time] = compute_permutation(model, X_subset, y_subset, scoring = 'roc_auc')

    # [shap_values, shap_time] = compute_shap(model, X_subset)
    [kernelshap_500_values, kernelshap_500_time] = compute_kernelshap(model, X_subset, samples=500)  # different values per run
    [kernelshap_1000_values, kernelshap_1000_time] = compute_kernelshap(model, X_subset, samples=1000)  # different values per run
    [marginalsage_1000_values, marginalsage_1000_time] = compute_sage(model, X_subset, y_subset, samples=1000, removal='marginal', binary_outcome=True)  # different values per run
    [conditionalsage_values, conditionalsage_time] = compute_sage(model, X_subset, y_subset, removal='surrogate', binary_outcome=True)  # different values per run
    # [sage_1000_values, sage_1000_time] = compute_sage(model, X_subset, y_subset, samples=1000, binary_outcome=True)  # different values per run

    feature_importance = pd.DataFrame({'coefficient': np.squeeze(coef_model),
                                       'permutation_auc': permutation_auc_values,
                                       'permutation_rmse': permutation_rmse_values,
                                       'permutation_balanced': permutation_balanced_values,
                                       'permutation_auc': permutation_auc_standard_values,
                                       # 'shap': shap_values.mean(axis=0).values,
                                       'kernelshap_500': kernelshap_500_values.mean(axis=0),
                                       'kernelshap_1000': kernelshap_1000_values.mean(axis=0),
                                       'marginalsage_1000': marginalsage_1000_values.values,
                                       'conditionalsage_1000': conditionalsage_values.values,
                                       # 'sage_1000': sage_1000_values.values
                                       })

    times = times.append(pd.Series({'name': re.search("data\d+", data).group(0),
                                    'permutation_auc': permutation_auc_time,
                                    'permutation_rmse': permutation_rmse_time,
                                    'permutation_balanced': permutation_balanced_time,
                                    'permutation_auc': permutation_auc_standard_time,
                                    # 'shap': shap_time,
                                    'kernelshap_500': kernelshap_500_time,
                                    'kernelshap_1000': kernelshap_1000_time,
                                    'marginalsage_1000': marginalsage_1000_time,
                                    'conditionalsage_1000': conditionalsage_time,
                                    # 'sage_1000': sage_1000_time
                                    }, dtype='object'), ignore_index=True)

    # sum across all feature importances
    # all_feature_importance = all_feature_importance + feature_importance

    # save feature importance
    feature_importance.to_csv(save_folder / "feature_importance" / str(re.sub('.pkl', '', m) + "_" + str(i) + "_feature_importance.csv"))
    # i=i+1

    # save computation times
    times.to_csv(save_folder / "computation_times.csv", index=False)
    print("end FI loop")

# take average
# feature_importance = all_feature_importance / i

# 5) Evaluate (dis)agreement
all_feature_importance = os.listdir(save_folder / "feature_importance")

results = pd.DataFrame()
metrics = ['overlap', 'rank', 'sign', 'ranksign', 'pearson', 'kendalltau', 'pairwise_comp', 'mae', 'rmse',
           'r2']

for f in all_feature_importance:
    feature_importance = pd.read_csv(save_folder / "feature_importance" / f, index_col=0)

    # Calculate evaluation metrics
    for i in range(feature_importance.shape[1]):
        for j in range(i):
            fi_meth1 = feature_importance.columns[i]
            fi_meth2 = feature_importance.columns[j]

            eval_metrics = fi_evaluate(fi1=feature_importance[fi_meth1], fi2=feature_importance[fi_meth2],
                                       metrics=metrics)

            # np.array(fi_meth1, fi_meth2)
            info = [re.search("data\d+", f).group(0), re.search("model-[a-z]+", f).group(0), fi_meth1, fi_meth2]
            results = results.append(pd.Series(info + eval_metrics, dtype='object'), ignore_index=True)

    # Visualize feature importance  # TODO: remove -> move all to dash explorer
    # plot = fi_visualize(feature_importance, output_folder=save_folder)

# Add names to columns
results.columns = ['name', 'model', 'fi_meth1', 'fi_meth2', ] + metrics
results.to_csv(save_folder / "final_evaluation.csv", index=False)

# 6) Output
# TODO: combine all output in a format to process