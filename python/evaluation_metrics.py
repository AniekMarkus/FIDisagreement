
# Modules
import numpy as np
import pandas as pd
import os

import itertools
from scipy.stats import pearsonr, rankdata
from scipy.special import comb
from scipy.stats import kendalltau
import sklearn
import matplotlib.pyplot as plt

# Get functions in other Python scripts
from help_functions import *

# code based on evaluation metrics OpenXAI: https://github.com/AI4LIFE-GROUP/OpenXAI

def fi_evaluate(fi1, fi2, metrics):
    """

     :param fi1: np.array, n x p
     :param fi2: np.array, n x p
     :param metric: choose from ...
     :return:
     """

    attrA = np.array(fi1).reshape(1, -1)
    attrB = np.array(fi2).reshape(1, -1)

    eval_metrics = []
    for metric in metrics:
        if metric in ('overlap', 'rank', 'sign', 'ranksign'):
            metric_distr = agreement_fraction(attrA, attrB, metric, k=5) # k=10
        elif metric in ('pearson', 'kendalltau'):
            metric_distr = rankcorr(attrA, attrB, metric)
        elif metric == 'pairwise_comp':
            metric_distr = pairwise_comp(attrA, attrB)
        elif metric in ('mae', 'rmse', 'r2'):
            metric_distr = error_calculation(attrA, attrB, metric)

        eval_metrics.append(np.mean(metric_distr))

    return eval_metrics

def agreement_fraction(attrA, attrB, metric, k=3):
    # id of top-k features
    topk_idA = np.argsort(-np.abs(attrA), axis=1)[:, 0:k]
    topk_idB = np.argsort(-np.abs(attrB), axis=1)[:, 0:k]

    # rank of top-k features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
    all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1) # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)
    topk_ranksA = np.take_along_axis(all_feat_ranksA, topk_idA, axis=1)
    topk_ranksB = np.take_along_axis(all_feat_ranksB, topk_idB, axis=1)

    # sign of top-k features
    topk_signA = np.take_along_axis(np.sign(attrA), topk_idA, axis=1)  # pos=1; neg=-1
    topk_signB = np.take_along_axis(np.sign(attrB), topk_idB, axis=1)

    # overlap agreement = (# topk features in common)/k
    if metric == 'overlap':
        topk_setsA = [set(row) for row in topk_idA]
        topk_setsB = [set(row) for row in topk_idB]
        # check if: same id
        metric_distr = np.array([len(setA.intersection(setB))/k for setA, setB in zip(topk_setsA, topk_setsB)])

    # rank agreement
    elif metric == 'rank':
        topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
        topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
        topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
        topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)

        #check if: same id + rank
        topk_id_ranksA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df)
        topk_id_ranksB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df)
        metric_distr = (topk_id_ranksA_df == topk_id_ranksB_df).sum(axis=1).to_numpy()/k

    # sign agreement
    elif metric == 'sign':
        topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id (contains rank info --> order of features in columns)
        topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
        topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
        topk_signB_df = pd.DataFrame(topk_signB).applymap(str)

        # check if: same id + sign
        topk_id_signA_df = ('feat' + topk_idA_df) + ('sign' + topk_signA_df)  # id + sign (contains rank info --> order of features in columns)
        topk_id_signB_df = ('feat' + topk_idB_df) + ('sign' + topk_signB_df)
        topk_id_signA_sets = [set(row) for row in topk_id_signA_df.to_numpy()]  # id + sign (remove order info --> by converting to sets)
        topk_id_signB_sets = [set(row) for row in topk_id_signB_df.to_numpy()]
        metric_distr = np.array([len(setA.intersection(setB))/k for setA, setB in zip(topk_id_signA_sets, topk_id_signB_sets)])

    # rank and sign agreement
    elif metric == 'ranksign':
        topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
        topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
        topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
        topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
        topk_signA_df = pd.DataFrame(topk_signA).applymap(str)  # sign
        topk_signB_df = pd.DataFrame(topk_signB).applymap(str)

        # check if: same id + rank + sign
        topk_id_ranks_signA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df) + ('sign' + topk_signA_df)
        topk_id_ranks_signB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df) + ('sign' + topk_signB_df)
        metric_distr = (topk_id_ranks_signA_df == topk_id_ranks_signB_df).sum(axis=1).to_numpy()/k

    else:
        raise NotImplementedError("Metric not implemented.")

    return metric_distr


def rankcorr(attrA, attrB, metric):
    corrs = []
    # rank features (accounting for ties)
    # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
    all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1)
    all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)

    for row in range(attrA.shape[0]):
        # Calculate correlation on ranks (iterate through rows: https://stackoverflow.com/questions/44947030/how-to-get-scipy-stats-spearmanra-b-compute-correlation-only-between-variable)
        if metric == "pearson":  # previous name: rankcorr!
            rho, _ = pearsonr(all_feat_ranksA[row, :], all_feat_ranksB[row, :])
        elif metric == "kendalltau":
            rho, _ = kendalltau(all_feat_ranksA[row, :], all_feat_ranksB[row, :])
        corrs.append(rho)

    # return metric's distribution and average
    return np.array(corrs)


def pairwise_comp(attrA, attrB):
        n_datapoints = attrA.shape[0]
        n_feat = attrA.shape[1]

        # rank of all features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
        # rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1)
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)

        # count # of pairs of features with same relative ranking
        feat_pairs_w_same_rel_rankings = np.zeros(n_datapoints)

        for feat1, feat2 in itertools.combinations_with_replacement(range(n_feat), 2):
            if feat1 != feat2:
                rel_rankingA = all_feat_ranksA[:, feat1] < all_feat_ranksA[:, feat2]
                rel_rankingB = all_feat_ranksB[:, feat1] < all_feat_ranksB[:, feat2]
                feat_pairs_w_same_rel_rankings += rel_rankingA == rel_rankingB

        pairwise_distr = feat_pairs_w_same_rel_rankings/comb(n_feat, 2)

        return pairwise_distr

def error_calculation(attrA, attrB, metric):
    # attrA = predicted, attrB = real
    attrA = normalise(attrA)
    attrB = normalise(attrB)

    if metric == 'mae':  # mean absolute error, best value = 0
        return np.abs(attrA - attrB).mean()
    elif metric == 'rmse':  # root mean squared error, best value = 0
        return np.sqrt(np.square(attrA - attrB).mean())
    elif metric == 'r2':  # (coefficient of determination) regression score function, best value = 1
        return sklearn.metrics.r2_score(attrA.transpose(), attrB.transpose())

    # TODO: does not work for distribution like other functions?

def fi_visualize(feature_importance, output_folder, sorted_on="permutation"):
    # Take absolute value
    abs_values = feature_importance.apply(lambda c: c.abs(), axis=0)

    # Scale values
    values_norm = abs_values.apply(lambda c: normalise(c), axis=0)

    values_norm = values_norm.sort_values(sorted_on, inplace=False, ascending=False, ignore_index=False)
    # coefficients_norm.index = coefficients_norm.index+1
    values_norm.plot(kind="line")
    plt.title('Visualization of feature importance / coefficients (normalized)')
    # plt.xticks(rotation=90)
    plt.xlabel("Variables (ordered from high to low model coefficient)")
    plt.ylabel("Importance (normalized)")
    # plt.xticks(range(0, 50, 5))

    rank = abs_values.apply(lambda c: c.rank(ascending=False), axis=0)
    # rank.index = X.columns
    rank = rank.sort_values(sorted_on, inplace=False, ascending=True, ignore_index=False)
    rank.to_csv(output_folder / 'ranking_abs_scaled.csv')

    return plt.show()