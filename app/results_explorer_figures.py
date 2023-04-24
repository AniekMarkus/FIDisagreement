import colorlover as cl
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
import os
import re
import seaborn as sns  # correlogram
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly

from python.help_functions import *
from app.results_explorer_helpers import *

import plotly.io as pio
pio.renderers.default = "browser"

# fig = go.Figure(data=go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[10, 11, 12, 13],
#     mode='markers',
#     marker=dict(size=[40, 60, 80, 100],
#                 color=[0, 1, 2, 3])
# ))
#
# fig.show()

def serve_fi_visualization(output_folder, color_dict,  dataset = "data1", fimethod = ["coefficient", "permutation_auc"], sorted_on="coefficient"):
    feature_importance = get_fi(output_folder, dataset)

    plot_data = feature_importance.loc[:, feature_importance.columns.isin(fimethod + ['variable'])]
    # plot_data = plot_data.groupby(by='variable', group_keys=True, as_index=False).mean()

    # Take absolute value
    # plot_data = plot_data.apply(lambda c: c.abs(), axis=0)

    # Scale values
    plot_data.loc[:, fimethod] = plot_data.loc[:, fimethod].apply(lambda c: normalise(c), axis=0)
    plot_data = plot_data.sort_values(sorted_on, inplace=False, ascending=False, ignore_index=False)
    # plot_data.variable = plot_data.variable.astype(str)

    order_variable = plot_data.variable

    # Translate wide to long format
    plot_data = pd.melt(plot_data, id_vars="variable", value_vars=fimethod, var_name="method")

    df_grouped = (plot_data.groupby(by=['variable', 'method']).agg(['mean', 'std', 'count']))
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    # Calculate a confidence interval as well.
    # df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    # df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    # df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    # df_grouped.head()

    df_grouped['colors'] = df_grouped['method'].map(color_dict)

    layout = go.Layout(
        title="Visualization of feature importance / coefficients (normalized)",
        xaxis=dict(title="Variables (ordered from high to low model coefficient)", gridcolor="#2f3445"),
        yaxis=dict(title="Importance (normalized)", gridcolor="#2f3445"),
        # legend=dict(x=0, y=1.05, orientation="v"),
        margin=dict(l=100, r=10, t=50, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )
    figure = go.Figure(layout=layout)
    for m in fimethod:
        values = df_grouped.loc[df_grouped.method == m, :]
        values = values.iloc[order_variable, :].reset_index(drop=True)
        values.variable = values.variable.astype(str)

        figure.add_traces([
                go.Scatter(
                    name=m,
                    x=values['variable'],
                    y=values['mean'],  # round(values['mean'], 2),
                    mode='lines',
                    line=dict(color=values.loc[0, 'colors']), # 'rgb(31, 119, 180)'
                )
                #,
                # go.Scatter(
                #     x=list(values['variable'])+list(values['variable'][::-1]), # x, then x reversed
                #     y=list(values['ci_upper'])+list(values['ci_lower'][::-1]), # upper, then lower reversed
                #     fill='toself',
                #     fillcolor=values.loc[0, 'colors'], #'rgba(0,100,80,0.2)',
                #     line=dict(color=values.loc[0, 'colors']),  # 'rgba(255,255,255,0)'
                #     opacity=0.3,
                #     hoverinfo='skip',
                #     showlegend=False,
                #     name='95% CI'
                # )
        ])

    # fig.update_layout(
    #     xaxis_title='Pickup Date',
    #     yaxis_title='Avg Fare',
    #     title='Avg Taxi Fare by Date',
    #     hovermode='x'
    # )
    # fig.update_yaxes(rangemode='tozero')

    return figure

def fi_ranking(output_folder, dataset="data3", fimethod=["permutation_auc", "coefficient"]):
    feature_importance = get_fi(output_folder, dataset)

    feature_importance = feature_importance.loc[:, feature_importance.columns.isin(fimethod + ['variable'])]

    # Take mean across iterations
    feature_importance = feature_importance.groupby(by='variable', group_keys=True, as_index=False).mean()

    # Take absolute value
    abs_values = feature_importance.loc[:, feature_importance.columns != 'variable'].apply(lambda c: c.abs(), axis=0)

    rank = abs_values.apply(lambda c: c.rank(ascending=False, method='min'), axis=0)
    rank['variable'] = rank.index
    rank = pd.melt(rank, id_vars="variable", var_name="method")

    # order_variable = ["coefficient", "sage_500", "sage_1000", "conditionalsage", "marginalsage_1000", "kernelshap_500", "kernelshap_1000",  "kernelshap_3000", "permutation_auc", "permutation_rmse"]
    # rank['method'] = rank['method'].astype(pd.api.types.CategoricalDtype(categories=order_variable, ordered=True))
    rank.sort_values(by="method", inplace=True)

    layout = go.Layout(
        title="Slope chart ranking (each line = variable)",
        # xaxis=dict(title="FI methods", gridcolor="#2f3445"),
        # yaxis=dict(title="Ranking", gridcolor="#2f3445"),
        xaxis=dict(title="Feature importance methods"),
        yaxis=dict(title="Rank position"),
        legend=None,
        margin=dict(l=100, r=10, t=50, b=40),
        # colorscale=plotly.graph_objs.layout.Colorscale(diverging='blues')
        # plot_bgcolor="#282b38",
        # paper_bgcolor="#282b38",
        # font={"color": "#a5b1cd"},
    )

    figure = go.Figure(layout=layout)
    for v in rank.variable:  # v = variables[0]
        temp = rank[rank.variable == v]

        # temp = temp.iloc[order_variable, :].reset_index(drop=True)

        figure.add_traces([
            go.Scatter(
                name=v,
                x=temp.method,
                y=temp.value,
                mode='lines'
                # line=dict(color=list_color[v])
            )])

    figure.update_layout(showlegend=False)
    # rank = rank.sort_values(sorted_on, inplace=False, ascending=True, ignore_index=False)
    # rank.to_csv(output_folder / 'ranking_abs_scaled.csv')

    return figure

def fi_topfeatures(output_folder, color_dict, dataset="data2", fimethod=["permutation", "shap"], k=10):
    feature_importance = get_fi(output_folder, dataset)

    feature_importance = feature_importance.loc[:, feature_importance.columns.isin(fimethod + ['variable'])]

    # TODO: test order?
    # Take mean across iterations
    feature_importance = feature_importance.groupby(by='variable', group_keys=True, as_index=False).mean()

    # Scale values (not necessary for ranking, but added to visualization later)
    # feature_importance[fimethod] = feature_importance[fimethod].apply(lambda c: normalise(c), axis=0)

    # Take absolute value
    abs_values = feature_importance.loc[:, feature_importance.columns != 'variable'].apply(lambda c: c.abs(), axis=0)

    rank = abs_values.apply(lambda c: c.rank(ascending=False, method='min'), axis=0)
    rank['variable'] = rank.index

    figure = make_subplots(1, len(fimethod), subplot_titles=fimethod)
    for i in range(1, len(fimethod)+1):
        m = fimethod[i-1]
        rank_i = rank.loc[:, [m, 'variable']]
        rank_i = rank_i.sort_values(m, inplace=False, ascending=True, ignore_index=True)
        rank_i = rank_i.iloc[:k, :]

        rank_i = pd.merge(rank_i, feature_importance.loc[:, [m, 'variable']], on='variable', suffixes=("_rank", ""))
        rank_i.variable = "var " + rank_i.variable.astype(str)

        rank_i = rank_i.iloc[::-1]  # reverse dataframe
        # figure.add_trace(go.Bar(x=rank.loc[:, fimethod[i-1]], y=rank.variable, orientation='h'), 1, i)
        figure.add_trace(go.Bar(x=rank_i.loc[:, m], y=rank_i.variable, orientation='h', name=m, marker_color=color_dict[m]), 1, i)
        # figure.update_layout(yaxis=dict(autorange="reversed"))

    # figure.update_xaxes(matches='x')
    figure.update_layout(title="Most important features",
                         barmode='stack',
                         showlegend=False)
    figure.update_annotations(font_size=10)

    figure.update_xaxes(visible=False)
    return figure


def serve_fi_metrics(color_dict, final_evaluation, dataset="data3", model="model-logistic", fimethod=["coefficient"], metrics=["overlap", "mae"]):
    plot_data = final_evaluation.loc[(final_evaluation.name == dataset) & (final_evaluation.model == model) &
                                     (final_evaluation.fi_meth1.isin(fimethod) | final_evaluation.fi_meth2.isin(fimethod)), :]

    # metrics_data=plot_data.loc[:, ~plot_data.columns.isin(['data', 'model', 'fi_meth1', 'fi_meth2', 'fi_meth'])]
    metrics_data=plot_data.loc[:, plot_data.columns.isin(metrics)]
    cols = metrics_data.columns.isin(['mae', 'rmse', 'r2'])
    metrics_data.loc[:, cols]=metrics_data.loc[:,cols].apply(lambda c: 1-normalise(c), axis=0)
    metrics_data.index=plot_data.fi_meth
    metrics_data[metrics_data == 0] = 0.01

    # Change names cols
    # metrics_data.columns = ["Top-5", "Sign agreement", "Kendall's tau", "1-MAE"]

    # Translate wide to long format
    metrics_data = pd.melt(metrics_data, value_vars=metrics, var_name="metrics", ignore_index=False)

    layout = go.Layout(
        title="Metrics",
        # xaxis=dict(title="Metrics", gridcolor="#2f3445"),
        # xaxis=dict(title="Metrics"),
        yaxis=dict(title="Value"),
        # yaxis=dict(title="Value (higher more agreement)", gridcolor="#2f3445"),
        # legend=dict(x=0, y=1.2, orientation="v"),
        margin=dict(l=100, r=10, t=25, b=40)
        # plot_bgcolor=  "#282b38",
        # paper_bgcolor="#282b38",
        # font={"color": "#a5b1cd"},
    )

    # Grouped box plot
    iterate = [i for i in metrics_data.index.unique() if "_coefficient" in i]

    figure = go.Figure(layout=layout)
    for row in iterate:  # row = metrics_data.index[0]
        figure.add_trace(go.Box(
            x=metrics_data.loc[metrics_data.index == row, "metrics"],
            y=metrics_data.loc[metrics_data.index == row, "value"],
            name=re.sub(pattern="_coefficient", repl="", string=row), # TODO: change permanent for all methods (e.g. using split)
            marker_color=color_dict[re.sub(pattern="_coefficient", repl="", string=row)] # TODO: change permanent for all methods (e.g. using split)
        ))

    # for row in metrics_data.metrics.unique():  # row = metrics_data.index[0]
    #     figure.add_trace(go.Box(
    #         x=metrics_data.index,
    #         y=metrics_data.loc[metrics_data.metrics == row, "value"],
    #         name=row
    #     ))

    figure.update_yaxes(range=[-0.1, 1.1])

    figure.update_layout(boxmode="group", margin=dict(l=100, r=10, t=25, b=150))
    figure.add_annotation(font=dict(size=10), x=0, y=-0.5, text="Note: higher values indicate more agreement.",
                          showarrow=False, textangle=0, xanchor='left', xref="paper", yref="paper")

    return figure


def serve_complexity_plot(output_folder, combined, list_characteristics, list_metrics, characteristic="rho", metric="overlap"):

    remove_cols = list(list_characteristics) + list(list_metrics)
    remove_cols.remove(characteristic)
    remove_cols.remove(metric)

    plot_data = combined.loc[:, ~combined.columns.isin(remove_cols)]

    df_grouped = plot_data.groupby(by=[characteristic, 'fi_meth'], group_keys=True, as_index=False).agg(['mean', 'std', 'count'])
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    # Calculate a confidence interval as well.
    df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    # df_grouped.head()

    cols = df_grouped.fi_meth.unique()

    # plot_data_wide = df_grouped.pivot(index=characteristic, columns='fi_meth', values=metric).reset_index()

    layout = go.Layout(
        title="...",
        xaxis=dict(title="Effect of varying data complexity", gridcolor="#2f3445"),
        yaxis=dict(title="Level of (dis)agreement", gridcolor="#2f3445"),
        # legend=dict(x=0, y=1.05, orientation="v"),
        margin=dict(l=100, r=10, t=50, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )
    figure = go.Figure(layout=layout)
    for col in cols: # col = cols[0]
        # figure.add_trace(go.Scatter(x=plot_data_wide[characteristic],
        #                             y=plot_data_wide[col],
        #                             name=col,
        #                             mode='lines',
        #                             line=dict(shape='linear'),
        #                             connectgaps=True
        #                             )
        #                  )

        values = df_grouped.loc[df_grouped.fi_meth == col, :].reset_index(drop=True)

        figure.add_traces([
                go.Scatter(
                    name=col,
                    x=values[characteristic],
                    y=values['mean'],  # round(values['mean'], 2),
                    mode='lines',
                    # line=dict(color=values.loc[0, 'colors']), # 'rgb(31, 119, 180)'
                ),
                go.Scatter(
                    x=list(values[characteristic])+list(values[characteristic][::-1]), # x, then x reversed
                    y=list(values['ci_upper'])+list(values['ci_lower'][::-1]), # upper, then lower reversed
                    fill='toself',
                    # fillcolor=values.loc[0, 'colors'], #'rgba(0,100,80,0.2)',
                    # line=dict(color=values.loc[0, 'colors']),  # 'rgba(255,255,255,0)'
                    opacity=0.3,
                    hoverinfo='skip',
                    showlegend=False,
                    name='95% CI'
                )])

    return figure


def serve_prediction_plot(
        model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold
):
    # Get train and test score from model
    y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscale
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        [0.0000000, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0000000, "#20e6ff"],
    ]

    # Create the plot
    # Plot the prediction contour of the SVM
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.9,
    )

    # Plot the threshold
    trace1 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo="none",
        contours=dict(
            showlines=False, type="constraint", operation="=", value=scaled_threshold
        ),
        name=f"Threshold ({scaled_threshold:.3f})",
        line=dict(color="#708090"),
    )

    # Plot Training Data
    trace2 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=10, color=y_train, colorscale=bright_cscale),
    )

    # Plot Test Data
    trace3 = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(
            size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale
        ),
    )

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        # legend=dict(x=0, y=-0.01, orientation="v"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0, trace1, trace2, trace3]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_roc_curve(model, X_test, y_test):
    decision_test = model.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace0 = go.Scatter(
        x=fpr, y=tpr, mode="lines", name="Test Data", marker={"color": "#13c6e9"}
    )

    layout = go.Layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        # legend=dict(x=0, y=1.05, orientation="v"),
        margin=dict(l=100, r=10, t=25, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_pie_confusion_matrix(model, X_test, y_test, Z, threshold):
    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    y_pred_test = (model.decision_function(X_test) > scaled_threshold).astype(int)

    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo="label+value+percent",
        textinfo="text+value",
        text=labels,
        sort=False,
        marker=dict(colors=colors),
        insidetextfont={"color": "white"},
        rotation=90,
    )

    layout = go.Layout(
        title="Confusion Matrix",
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def data_correlogram(output_folder, dataset):
    # find all patterns
    file = os.listdir(output_folder / "data")
    file = list(filter(lambda v: re.findall(dataset, v), file))
    file = list(filter(lambda v: re.findall("_Xtrain.csv", v), file))

    X = pd.read_csv(output_folder / "data" / file[0])

    # Compute correlations
    # corr = X.drop('intercept', axis=1).corr()
    corr = X.corr()

    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    #
    #
    # plt.title('Correlogram of features')
    # return plt.show()

    # colorscale = [[0.0, "rgb(165,0,38)"],
    #               [0.5, "rgb(165,0,38)"]
    #               [1.0, "rgb(49,54,149)"]]
    figure = go.Figure(data=go.Heatmap(z=corr, zmin=-1, zmax=1, colorscale='RdBu', hoverinfo='text', text=corr))

    return figure

