
# Modules
import numpy as np
import pandas as pd
import os

import random
from scipy.io import arff
from sklearn import preprocessing

# Get functions in other Python scripts


os.environ["CUDA_VISIBLE_DEVICES"]=""

# Get functions in other Python scripts
from models import *

def get_data(data, repeat, root_folder, output_folder, rerun):
    data_folder = output_folder / "data"

    if rerun:
        # Process data
        if data in ['dgp1']:
            X, y = simulate_data(data, root_folder, settings_file='dgp_specs.csv')
        else:
            X, y = load_data(data, root_folder)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Data normalization (minmax)
        X_train = pd.DataFrame(preprocessing.minmax_scale(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(preprocessing.minmax_scale(X_test), columns=X_test.columns)

        # Save data
        X_train.to_csv(data_folder / str(f"{data}-{repeat}-Xtrain.csv"), index=False)
        pd.DataFrame(y_train).to_csv(data_folder / str(f"{data}-{repeat}-ytrain.csv"), index=False)

        X_test.to_csv(data_folder / str(f"{data}-{repeat}-Xtest.csv"), index=False)
        pd.DataFrame(y_test).to_csv(data_folder / str(f"{data}-{repeat}-ytest.csv"), index=False)

    else:
        # Load data
        X_train = pd.read_csv(data_folder / str(f"{data}-{repeat}-Xtrain.csv"))
        y_train = pd.read_csv(data_folder / str(f"{data}-{repeat}-ytrain.csv"))

        X_test = pd.read_csv(data_folder / str(f"{data}-{repeat}-Xtest.csv"))
        y_test = pd.read_csv(data_folder / str(f"{data}-{repeat}-ytest.csv"))

    print("> get_data done")
    return X_train, X_test, y_train, y_test


def simulate_data(data, root_folder, settings_file):
    # Input
    input_settings = pd.read_csv(root_folder + '/input-settings/' + settings_file, index_col=0, sep=';')

    # Generate data using input parameters
    dgp = input_settings.loc[input_settings.name == data]

    # TODO: check this function
    [X, y] = simulate(N=dgp.N.values,
                      o=dgp.o.values,
                      beta=dgp.beta.values,
                      F=dgp.F.values,
                      inf=dgp.inf.values,
                      t=dgp.t.values,
                      rho=dgp.rho.values,
                      e=dgp.e.values,
                      A=dgp.A.values,
                      L=dgp.L.values,
                      seed=2023)
    return X, y

def load_data(data, root_folder):
    # TODO: give specs data format needs to satisfy

    # Input
    input_data = arff.loadarff(root_folder + "/input-data/" + data + ".arff")
    input_data = pd.DataFrame(input_data[0])

    # Split data
    X = input_data[input_data.columns[~input_data.columns.isin(['class'])]]
    y = input_data['class']

    return X, y

def modify_data(data, output_folder, params, X_train, X_test, y_train, y_test, rerun):
    data_folder = output_folder / "data"

    if rerun:
        change = False
        if params != '':
            # Modify data
            if params['change'] == "baseline":
                # For version v0 do continue analysis
                change=True

            elif params['change'] == "num_features":
                if params['method'] == "random":
                    # Randomly select subset of features (change X data)
                    selected_vars = random.sample(range(X_train.shape[1]), min(params["value"], X_train.shape[1]))

                    # Check if change occurred
                    if len(selected_vars) < X_train.shape[1]:
                        change = True

                    X_train = X_train.iloc[:, selected_vars]
                    X_test = X_test.iloc[:, selected_vars]

            elif params['change'] == "num_obs":
                if params['method'] == "random":
                    # Randomly select subset of observations (change train set)
                    selected_rows = random.sample(range(X_train.shape[0]), min(params["value"], X_train.shape[0]))

                    # Check if change occurred
                    if len(selected_rows) < X_train.shape[0]:
                        change = True

                    X_train = X_train.iloc[selected_rows]
                    y_train = y_train.iloc[selected_rows]

            elif params['change'] == "num_outcomes":
                if params['method'] == "random":
                    # Reset index
                    X_train.reset_index(drop=True, inplace=True)
                    y_train.reset_index(drop=True, inplace=True)

                    # Randomly select subset of outcomes while keeping non-outcomes (change train set)
                    rows_zeros = y_train.index[y_train == 0]
                    rows_ones = y_train.index[y_train == 1]

                    selected_ones = random.sample(list(rows_ones), min(params["value"], len(rows_ones)))
                    selected_rows = list(selected_ones) + list(rows_zeros)

                    # Check if change occurred
                    if len(selected_rows) < X_train.shape[0]:
                        change = True

                    X_train = X_train.iloc[selected_rows]
                    y_train = y_train.iloc[selected_rows]

            elif params['change'] == "correlation":
                # Remove features with correlation above certain threshold (change X data)
                if params['method'] == "univariate":
                    # TODO: implement scalable way to compute this
                    # cor_matrix = compute_correlation_custom(X_train)
                    cor_matrix = X_train.corr(method="spearman")

                    # Remove variance on diagonal + upper/lower triangle and take absolute value
                    diag_matrix = np.zeros((X_train.shape[1], X_train.shape[1]), int)
                    np.fill_diagonal(diag_matrix, 1)
                    cor_matrix = np.abs(cor_matrix - diag_matrix)

                    # Find features above threshold and remove remaining features for further computation
                    selected_vars = cor_matrix.index[(cor_matrix>params['value']).any()]
                    X = X_train.loc[:, selected_vars]

                    # Remove those one by one
                    removed_vars = []

                    while selected_vars.shape[0] > 0:
                        f = selected_vars[-1]  # Starting with last, assuming less important
                        X = X.drop(f, axis=1)
                        removed_vars.append(f)

                        # cor_matrix = compute_correlation_custom(X_train)
                        cor_matrix = X.corr(method="spearman")

                        # Find features above threshold and remove remaining features for further computation
                        selected_vars = cor_matrix.index[(cor_matrix>params['value']).any()]
                        X = X.loc[:, selected_vars]

                    # Check if change occurred
                    if len(removed_vars) > 0:
                        change = True

                    # Remove from X data
                    X_train = X_train.drop(removed_vars, axis=1)
                    X_test = X_test.drop(removed_vars, axis=1)

            elif params['change'] == "prev_features":
                # Randomly remove ones for features (change X train data)
                if params['method'] == "random":

                    # Iterate through each predictor
                    for predictor in X_train.columns:
                        var = X_train[predictor]

                        if ((var==0) | (var==1)).all(): # Check if binary variable
                            count = var.sum()
                            remove = np.ceil(count*params["value"])

                            selected_rows_p = random.sample(range(X_train.shape[0]), remove.astype(int))

                            # Check if change occurred
                            if len(selected_rows_p) > 0:
                                change = True

                            X_train[predictor].iloc[selected_rows_p] = 0  # remove records binary vars by imputing zeros

            # Save modified data if changed
            if change:
                X_train.to_csv(data_folder / str(f"{data}-Xtrain.csv"), index=False)
                pd.DataFrame(y_train).to_csv(data_folder / str(f"{data}-ytrain.csv"), index=False)

                X_test.to_csv(data_folder / str(f"{data}-Xtest.csv"), index=False)
                pd.DataFrame(y_test).to_csv(data_folder / str(f"{data}-ytest.csv"), index=False)
                print("> modify_data done")

        elif params == '':
            # For baseline run do continue analysis
            change=True
            print("> modify_data skip")

    else:
        fileName = data_folder / str(f"{data}-Xtrain.csv")

        if os.path.exists(fileName):
            # Load modified data
            X_train = pd.read_csv(fileName)
            y_train = pd.read_csv(data_folder / str(f"{data}-ytrain.csv"))

            X_test = pd.read_csv(data_folder / str(f"{data}-Xtest.csv"))
            y_test = pd.read_csv(data_folder / str(f"{data}-ytest.csv"))

            # If modified data exists continue analysis
            change=True
            print("> modify_data load")

        else:
            # Do not continue to model
            change=False
            print("> modify_data does not exist")

    return X_train, X_test, y_train, y_test, change


# TODO: think about reusing parts of the simulated data to resolve differences? or replicate, but how many times?

def simulate(N=1000, o='Binary', beta=[1, 1, 1, 1, 1], F=4, inf=0.8, t='Binary', rho=0.1, e='None', A='None', L='None', seed=2022):
    """
    Simulate dataset with properties according to the below parameters.

    :param N: Sample size / number of observations.
    :param o: Outcome type = [Binary, Continuous].
    :param beta: Model coefficients (first value = intercept).
    :param F: Number of features.
    :param inf: Percentage of informative features (number rounded up).
    :param t: Type of features = [Binary, Continuous, Mixed].
    :param rho: Correlation / dependency between features between 0-1.
    :param e: Variance of normal distributed error term (or None).
    :param A: Non-additivity (e.g. quadratic terms) e.g. [2] (or None).
    :param L: Non-linearity / interaction terms e.g. [[1,2]] (or None).
    :return:
    """

    # TODO: check inputs (e.g. length beta compatible with F, ranges rho / e / A / L)

    # Set seed
    np.random.seed(seed)

    # Simulate informative features
    X_M = pd.DataFrame()

    M = int(np.ceil(F * inf))  # number of informative features
    U = F - M  # number of uninformative features

    mean = 0  # normal distributed variables
    variance = 1  # normal distributed variables
    prob = 0.1  # bernoulli distributed variables

    # Add correlated features
    # TODO: Extend to more than first two features? (in this case add in loop below?)
    # TODO: What about binary features?
    if rho != 0:
        x_1 = np.random.normal(loc=0, scale=1, size=N)  # standard normal
        x_2 = np.random.normal(loc=0, scale=1, size=N)  # standard normal

        mu_x1 = mean
        sigma_x1 = variance
        mu_x2 = mean
        sigma_x2 = variance

        x1_corr = mu_x1 + sigma_x1 * x_1
        x2_corr = mu_x2 + sigma_x2 * (rho * x_1 + np.sqrt(1 - rho ** 2) * x_2)

        X_M = pd.concat([X_M, pd.DataFrame(x1_corr), pd.DataFrame(x2_corr)], axis=1)

    # Add remaining features
    for m in range(X_M.shape[1], M):  # Start after generation of correlated features
        if t == 'Continuous':
            # Generate normal random variable with mean mu and
            x_i = np.random.normal(loc=mean, scale=variance, size=N)
        elif t == 'Binary':
            # Generate binary random variable
            x_i = np.random.binomial(n=1, p=prob, size=N)
        elif t == 'Mixed':
            if m < M / 2:  # First half continuous
                x_i = np.random.normal(loc=mean, scale=variance, size=N)
            else:  # Second half binary
                x_i = np.random.binomial(n=1, p=prob, size=N)

        X_M = pd.concat([X_M, pd.DataFrame(x_i)], axis=1)

    # Add non-additive features (not adding anything for binary variables!)
    if A != 'None':
        for a in A:
            x_a = np.square(X_M.iloc[:, a])
            X_M = pd.concat([X_M, pd.DataFrame(x_a)], axis=1)
            beta = np.append(beta, 0.5) # TODO: let this vary as well?

    # Add non-linear features
    if L != 'None':
        for l in L:
            x_l = X_M.iloc[:, l[0]] * X_M.iloc[:, l[1]]
            X_M = pd.concat([X_M, pd.DataFrame(x_l)], axis=1)
            beta = np.append(beta, 0.5)  # TODO: let this vary as well?

    # Simulate outcome
    z = beta[0] + X_M.dot(beta[1:M+1])

    # Add random error term / noise
    if e != 'None':
        z = z + np.random.normal(loc=0, scale=e, size=N)

    if o == 'Continuous':
        y = z
    elif o == 'Binary':
        pr = 1 / (1 + np.exp(-z))
        y = np.random.binomial(1, pr, N)

    # Simulate uninformative features
    X_U = pd.DataFrame()

    # TODO: random draw parameters?
    mean = 0  # normal distributed variables
    variance = 1  # normal distributed variables
    prob = 0.1  # bernoulli distributed variables

    for u in range(U):
        if t == 'Continuous':
            # Generate normal random variable with mean mu and
            u_i = np.random.normal(loc=mean, scale=variance, size=N)
        elif t == 'Binary':
            # Generate binary random variable
            u_i = np.random.binomial(n=1, p=prob, size=N)
        elif t == 'Mixed':
            if u < U / 2:  # First half continuous
                u_i = np.random.normal(loc=mean, scale=variance, size=N)
            else:  # Second half binary
                u_i = np.random.binomial(n=1, p=prob, size=N)

        X_U = pd.concat([X_U, pd.DataFrame(u_i)], axis=1)

    # Return simulated data
    X = pd.concat([X_M, X_U], axis=1)

    print('Number of outcomes:' + str(sum(y)) + ", outcome rate: " + str(sum(y) * 100.0 / N))

    return X, y
