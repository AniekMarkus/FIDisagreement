# Modules
import numpy as np
import pandas as pd
import os

# Get functions in other Python scripts

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
