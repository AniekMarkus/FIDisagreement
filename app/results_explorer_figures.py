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

def fi_values(output_folder,
              color_dict,
              dataset = "iris",
              version = "v0",
              model = "logistic",
              fimethod = ["permutation_auc"],
              sorted_on="None"):

    # Get feature importance values
    fi_values, settings_data = get_fi(output_folder, dataset, version, model, fimethod, scale=True)

    # Sort values
    if sorted_on != "None":
        fi_values = fi_values.sort_values(sorted_on, inplace=False, ascending=False, ignore_index=False)

    # Combine data
    fi_values = combine_fi(fi_values, settings_data)

    # Group data
    plot_data = fi_values.groupby(by=['variable', 'fi_method']).agg(['mean', 'std', 'count'])
    plot_data = plot_data.droplevel(axis=1, level=0).reset_index()

    # Calculate confidence interval
    plot_data['ci'] = 1.96 * plot_data['std'] / np.sqrt(plot_data['count'])
    plot_data['ci_lower'] = plot_data['mean'] - plot_data['ci']
    plot_data['ci_upper'] = plot_data['mean'] + plot_data['ci']

    # Link to colors
    plot_data['colors'] = plot_data['fi_method'].map(color_dict)

    figure = go.Figure(layout=std_layout)
    for m in plot_data.fi_method.unique():  # m = "permutation_auc"
        values = plot_data.loc[plot_data.fi_method == m, :]
        # values = values.iloc[order_variable, :]
        values = values.reset_index(drop=True)
        values.variable = values.variable.astype(str)

        figure.add_traces([
                go.Scatter(
                    name=m,
                    x=values['variable'],
                    y=values['mean'],
                    mode='lines',
                    line=dict(color=values.loc[0, 'colors'])
                )
                # , go.Scatter(
                #     x=list(values['variable'])+list(values['variable'][::-1]),  # x, then x reversed
                #     y=list(values['ci_upper'])+list(values['ci_lower'][::-1]),  # upper, then lower reversed
                #     fill='toself',
                #     fillcolor=values.loc[0, 'colors'],
                #     line=dict(color=values.loc[0, 'colors']),
                #     opacity=0.3,
                #     hoverinfo='skip',
                #     showlegend=True,
                #     name='95% CI' # m
                # )
        ])

    figure.update_layout(
        title="Feature importance (line = FI method)",
        xaxis=dict(title="Variables"),
        yaxis=dict(title="Importance (scaled)"))

    figure.write_image(output_folder / "plots" / f'{dataset}-{version}-{model}-fi_values.svg',
                       width=500, height=500)

    return figure

def fi_ranking(output_folder,
               color_dict,
               dataset = "iris",
               version = ["v0"],
               model = ["logistic"],
               fimethod = ["permutation_auc"]):

    # Get feature importance ranking
    fi_rank, fi_values, settings_data = get_rank(output_folder, dataset, version, model, fimethod)

    # Combine data
    fi_rank = combine_fi(fi_rank, settings_data)

    # Take mean across iterations
    plot_data = fi_rank.groupby(by=['variable', 'fi_method'], group_keys=True, as_index=False).mean()

    figure = go.Figure(layout=std_layout)
    for v in plot_data.variable:
        temp = plot_data[plot_data.variable == v]

        figure.add_traces([
            go.Scatter(
                name=v,
                x=temp.fi_method,
                y=temp.value,
                mode='lines'
                # line=dict(color=list_color[v])
            )])

    figure.update_layout(
        title="Ranking (line = variable)",
        xaxis=dict(title="FI method"),
        yaxis=dict(title="Ranking"),
        showlegend=False)

    figure.write_image(output_folder / "plots" / f'{dataset}-{version}-{model}-fi_ranking.svg',
                       width=500, height=500)

    return figure

def fi_topfeatures(output_folder,
                   color_dict,
                   dataset = "iris",
                   version = ["v0"],
                   model = ["logistic"],
                   fimethod = ["permutation_auc"],
                   k=5):

    # Get feature importance ranking
    fi_rank, fi_values, settings_data = get_rank(output_folder, dataset, version, model, fimethod, scale=True)

    # Combine data
    fi_rank = combine_fi(fi_rank, settings_data)
    fi_values = combine_fi(fi_values, settings_data)

    # Take mean across iterations
    fi_rank = fi_rank.groupby(by=['variable', 'fi_method'], group_keys=True, as_index=False).mean()
    fi_values = fi_values.groupby(by=['variable', 'fi_method'], group_keys=True, as_index=False).mean()

    # Update which methods occur in data
    fimethod = fi_values.fi_method.unique()

    # Translate long to wide format
    fi_rank = pd.pivot(fi_rank, index='variable', columns='fi_method', values='value')
    fi_rank.reset_index(inplace=True)

    fi_values = pd.pivot(fi_values, index='variable', columns='fi_method', values='value')
    fi_values.reset_index(inplace=True)

    figure = go.Figure(layout=std_layout).set_subplots(1, len(fimethod), horizontal_spacing=0.1, subplot_titles=fimethod)
    for i in range(1, len(fimethod)+1):
        m = fimethod[i-1]

        rank_i = fi_rank.loc[:, [m, 'variable']]
        rank_i = rank_i.sort_values(m, inplace=False, ascending=True, ignore_index=True)
        rank_i = rank_i.iloc[:k, :]

        rank_i = pd.merge(rank_i, fi_values.loc[:, [m, 'variable']], on='variable', suffixes=("_rank", ""))
        rank_i.variable = "var " + rank_i.variable.astype(str)

        rank_i = rank_i.iloc[::-1]  # reverse dataframe
        figure.add_trace(go.Bar(x=rank_i.loc[:, m], y=rank_i.variable, orientation='h', name=m, marker_color=color_dict[m]), 1, i)
        # figure.update_layout(yaxis=dict(autorange="reversed"))

    figure.update_layout(title="Top features",
                         barmode='stack',
                         showlegend=False)
    figure.update_annotations(font_size=10)
    figure.update_xaxes(visible=False)

    figure.write_image(output_folder / "plots" / f'{dataset}-{version}-{model}-fi_topfeatures.svg',
                       width=500, height=500)

    return figure


def fi_metrics(output_folder,
               fig_name,
               color_dict,
               dataset = "iris",
               version = "v0",
               model = "logistic",
               fimethod = ["permutation_auc", "permutation_ba"],
               eval_metrics = ["overlap", "mae"],
               summarize = False):

    # Get feature importance metrics
    res_metrics, eval_names = get_metrics(output_folder, dataset, version, model, fimethod, eval_metrics, summarize)

    # For visualization purposes show small number instead of zero
    res_metrics[res_metrics == 0] = 0.01

    # Change names cols
    # TODO: create dictionary for names/labels (like colors)
    # metrics_data.columns = ["Top-5", "Sign agreement", "Kendall's tau", "1-MAE"]

    # Combine and translate wide to long format
    if summarize:
        metrics_data = pd.concat([eval_names, res_metrics.rename('disagreement')], axis=1)
        metrics_data = pd.melt(metrics_data, id_vars=["fi_method1", "fi_method2"], value_vars='disagreement', var_name="metrics", ignore_index=False)
    else:
        metrics_data = pd.concat([eval_names, res_metrics], axis=1)
        metrics_data = pd.melt(metrics_data, id_vars=["fi_method1", "fi_method2"], value_vars=eval_metrics, var_name="metrics", ignore_index=False)

    # Change names cols
    # metrics_data.replace({"fi_method1": FI_name_dict, "fi_method2": FI_name_dict},inplace=True)

    # Grouped box plot
    comparison = [fimethod[0]]
    metrics_data = metrics_data.loc[metrics_data.fi_method2.isin(comparison), :]
    iterate = metrics_data.fi_method1.unique()

    figure = go.Figure(layout=std_layout)
    for i in iterate:  # i = iterate[0]
        figure.add_trace(go.Box(
            x=metrics_data.loc[metrics_data.fi_method1 == i, "metrics"],
            y=metrics_data.loc[metrics_data.fi_method1 == i, "value"],
            name=i,
            marker_color=color_dict[i]
        ))

    figure.update_yaxes(range=[-0.1, 1.1])

    figure.update_layout(title="Evaluation disagreement",
                         xaxis=dict(title="Metrics"),
                         yaxis=dict(title="Agreement"),
                         boxmode="group",
                         margin=dict(l=100, r=10, t=25, b=150),
                         legend=dict(x=0, y=1.2, orientation="v"))

    figure.add_annotation(font=dict(size=10), x=0, y=-0.5, text="Note: higher values indicate more agreement.",
                          showarrow=False, textangle=0, xanchor='left', xref="paper", yref="paper")

    figure.write_image(output_folder / "plots" / f'{dataset}-{version}-{model}-fi_metrics-{fig_name}.svg',
                       width=500, height=500)

    return figure


def heatmap_disagreement(output_folder,
                         fig_name,
                         dataset = "iris",
                         version = "v0",
                         model = "logistic",
                         fimethod = ["permutation_auc", "permutation_ba"],
                         eval_metrics = ["overlap", "mae"]):

    # Need one output value
    if len(eval_metrics) > 1:
        summarize = True
    else:
        summarize = False

    # Get feature importance metrics
    res_metrics, eval_names = get_metrics(output_folder, dataset, version, model, fimethod, eval_metrics, summarize)

    # For visualization purposes show small number instead of zero
    res_metrics[res_metrics == 0] = 0.01

    # Combine and translate wide to long format
    metrics_data = pd.concat([eval_names, res_metrics.rename('disagreement')], axis=1)

    # Change names cols
    metrics_data.replace({"fi_method1": FI_name_dict, "fi_method2": FI_name_dict},inplace=True)
    metrics_data = pd.pivot(metrics_data, index="fi_method1", columns="fi_method2", values="disagreement")

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

    figure = go.Figure(data=go.Heatmap(z=metrics_data, zmin=0, zmax=1, colorscale='RdBu', texttemplate="%{text:.2f}",
                                       text=metrics_data, x=metrics_data.index, y=metrics_data.columns))

    figure.write_image(output_folder / "plots" / f'{dataset}-{version}-{model}-heatmap-{fig_name}.svg',
                       width=500, height=500)

    return figure


def complexity_plot(output_folder,
                    fig_name,
                    color_dict,
                    modify_params,
                    dataset="iris",
                    version="v2",
                    model="logistic",
                    fimethod=["permutation_auc", "permutation_ba"],
                    metrics="mae"):

    # Get feature importance metrics for all versions
    version_list = os.listdir(output_folder / "result")
    version_list = list(filter(lambda v: re.findall(dataset, v), version_list))
    version_list = list(filter(lambda v: re.findall(version, v), version_list))
    version_list = list(set(map(lambda v: v.split(sep="-")[1], version_list)))

    all_metrics = pd.DataFrame()

    for v in version_list:
        eval_metrics, eval_names = get_metrics(output_folder, dataset, v, model, fimethod, metrics, summarize=True)

        metrics_v = pd.concat([eval_names, eval_metrics.rename('disagreement')], axis=1)
        metrics_v["version"] = v

        all_metrics = all_metrics.append(metrics_v)

    # Take mean across fi methods
    plot_data = all_metrics.groupby(by=['fi_method1', 'version'], group_keys=True, as_index=False).mean()

    plot_data = pd.merge(plot_data, modify_params, on='version')

    figure = go.Figure(layout=std_layout)
    for fi in all_metrics.fi_method1.unique():
        values = plot_data.loc[plot_data.fi_method1 == fi, :].reset_index(drop=True)

        figure.add_traces([
            go.Scatter(
                name=fi,
                x=values.value,  # version
                y=values.disagreement,
                # mode='markers',
                line=dict(color=color_dict[fi])
            )])

    # Sort x axis by increasing values = more complexity

    figure.update_layout(# title=str("Effect of modifications " + version),
                         xaxis=dict(title="Increasing data complexity", tickmode='array', tickvals=values.value),
                         yaxis=dict(title="Agreement", range=[0,1]))

    figure.write_image(output_folder / "plots" / f'{dataset}-{version}-{model}-fi_complexity-{fig_name}.svg',
                       width=500, height=500)

    return figure

# def prediction_plot(
#         model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold
# ):
#     # Get train and test score from model
#     y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
#     y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
#     train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
#     test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)
#
#     # Compute threshold
#     scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
#     range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))
#
#     # Colorscale
#     bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
#     cscale = [
#         [0.0000000, "#ff744c"],
#         [0.1428571, "#ff916d"],
#         [0.2857143, "#ffc0a8"],
#         [0.4285714, "#ffe7dc"],
#         [0.5714286, "#e5fcff"],
#         [0.7142857, "#c8feff"],
#         [0.8571429, "#9af8ff"],
#         [1.0000000, "#20e6ff"],
#     ]
#
#     # Create the plot
#     # Plot the prediction contour of the SVM
#     trace0 = go.Contour(
#         x=np.arange(xx.min(), xx.max(), mesh_step),
#         y=np.arange(yy.min(), yy.max(), mesh_step),
#         z=Z.reshape(xx.shape),
#         zmin=scaled_threshold - range,
#         zmax=scaled_threshold + range,
#         hoverinfo="none",
#         showscale=False,
#         contours=dict(showlines=False),
#         colorscale=cscale,
#         opacity=0.9,
#     )
#
#     # Plot the threshold
#     trace1 = go.Contour(
#         x=np.arange(xx.min(), xx.max(), mesh_step),
#         y=np.arange(yy.min(), yy.max(), mesh_step),
#         z=Z.reshape(xx.shape),
#         showscale=False,
#         hoverinfo="none",
#         contours=dict(
#             showlines=False, type="constraint", operation="=", value=scaled_threshold
#         ),
#         name=f"Threshold ({scaled_threshold:.3f})",
#         line=dict(color="#708090"),
#     )
#
#     # Plot Training Data
#     trace2 = go.Scatter(
#         x=X_train[:, 0],
#         y=X_train[:, 1],
#         mode="markers",
#         name=f"Training Data (accuracy={train_score:.3f})",
#         marker=dict(size=10, color=y_train, colorscale=bright_cscale),
#     )
#
#     # Plot Test Data
#     trace3 = go.Scatter(
#         x=X_test[:, 0],
#         y=X_test[:, 1],
#         mode="markers",
#         name=f"Test Data (accuracy={test_score:.3f})",
#         marker=dict(
#             size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale
#         ),
#     )
#
#     layout = go.Layout(
#         xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
#         yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
#         hovermode="closest",
#         # legend=dict(x=0, y=-0.01, orientation="v"),
#         margin=dict(l=0, r=0, t=0, b=0),
#         plot_bgcolor="#282b38",
#         paper_bgcolor="#282b38",
#         font={"color": "#a5b1cd"},
#     )
#
#     data = [trace0, trace1, trace2, trace3]
#     figure = go.Figure(data=data, layout=layout)
#
#     return figure
#
#
# def roc_curve(model, X_test, y_test):
#     decision_test = model.decision_function(X_test)
#     fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)
#
#     # AUC Score
#     auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)
#
#     trace0 = go.Scatter(
#         x=fpr, y=tpr, mode="lines", name="Test Data", marker={"color": "#13c6e9"}
#     )
#
#     layout = go.Layout(
#         title=f"ROC Curve (AUC = {auc_score:.3f})",
#         xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
#         yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
#         # legend=dict(x=0, y=1.05, orientation="v"),
#         margin=dict(l=100, r=10, t=25, b=40),
#         plot_bgcolor="#282b38",
#         paper_bgcolor="#282b38",
#         font={"color": "#a5b1cd"},
#     )
#
#     data = [trace0]
#     figure = go.Figure(data=data, layout=layout)
#
#     return figure
#
#
# def pie_confusion_matrix(model, X_test, y_test, Z, threshold):
#     # Compute threshold
#     scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
#     y_pred_test = (model.decision_function(X_test) > scaled_threshold).astype(int)
#
#     matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
#     tn, fp, fn, tp = matrix.ravel()
#
#     values = [tp, fn, fp, tn]
#     label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
#     labels = ["TP", "FN", "FP", "TN"]
#     blue = cl.flipper()["seq"]["9"]["Blues"]
#     red = cl.flipper()["seq"]["9"]["Reds"]
#     colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]
#
#     trace0 = go.Pie(
#         labels=label_text,
#         values=values,
#         hoverinfo="label+value+percent",
#         textinfo="text+value",
#         text=labels,
#         sort=False,
#         marker=dict(colors=colors),
#         insidetextfont={"color": "white"},
#         rotation=90,
#     )
#
#     layout = go.Layout(
#         title="Confusion Matrix",
#         margin=dict(l=50, r=50, t=100, b=10),
#         legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
#         plot_bgcolor="#282b38",
#         paper_bgcolor="#282b38",
#         font={"color": "#a5b1cd"},
#     )
#
#     data = [trace0]
#     figure = go.Figure(data=data, layout=layout)
#
#     return figure


def data_correlogram(output_folder,
                     dataset="iris"):

    # Load original data
    X = pd.read_csv(output_folder / "data" / str(dataset + "-1-v0-Xtrain.csv"))

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


