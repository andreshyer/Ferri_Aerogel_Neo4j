import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd


def pva_graph(predictions_stats, predictions,run_name):
    """
    Make Predicted vs. Actual graph with prediction uncertainty.
    Pass dataframe from multipredict function. Return a graph.
    """
    # Reuse function for scaled data
#     if use_scaled:
#         pva = scaled_predictions
#     else:g
    pva = predictions

    r2 = predictions_stats['r2_raw']
    mse = predictions_stats['mse_raw']
    rmse = predictions_stats['rmse_raw']

    plt.rcParams['figure.figsize'] = [12, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    norm = cm.colors.Normalize(vmax=pva['pred_std'].max(), vmin=pva['pred_std'].min())
    x = pva['actual']
    y = pva['pred_avg']
    plt.scatter(x, y, c=pva['pred_std'], cmap='plasma', norm=norm, alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label("Uncertainty")

    # set axis limits
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
            ]

    # ax = plt.axes()
    plt.xlabel('True', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
#     if use_scaled:
#         plt.title(run_name + f' Predicted vs. Actual (scaled)')
#     else:
#         plt.title(run_name + f' Predicted vs. Actual')

    plt.plot(lims, lims, 'k-', label='y=x')
    plt.plot([], [], ' ', label='R^2 = %.3f' % r2)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    plt.plot([], [], ' ', label='MSE = %.3f' % mse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent

    plt.savefig(run_name + '_' + f'PVA.png')


def impgraph_tree_algorithm(algorithm, estimator, feature_list, run_name):
    """
    Objective: Make a feature importance graph. I'm limiting this to only rf and gdb since only they have feature
    importance (I might need to double check on that). I'm also limiting this to only rdkit2d since the rest are only 0s
    and 1s
    """
    if algorithm not in ["xgb", "nn"]:
        pass
    else:
        Exception("Don't use this feature importance function for " + algorithm)

    # Get numerical feature importances

    importances2 = estimator.feature_importances_  # used later for graph

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, list(importances2))]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
#     if len(feature_importances) > 100:
    feature_importances = feature_importances[:30]
    # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # prepare importance data for export and graphing
    # indicies = (-importances2).argsort()
    varimp = pd.DataFrame([], columns=['variable', 'importance'])
    varimp['variable'] = [feature_list[i] for i in feature_importances]
    varimp['importance'] = importances2[feature_importances]
    # Importance Bar Graph
    plt.rcParams['figure.figsize'] = [15, 9]

    # Set the style
    plt.style.use('bmh')
    varimp = varimp.head(30)
    # intiate plot (mwahaha)
    fig, ax = plt.subplots()
    plt.bar(varimp.index, varimp['importance'], orientation='vertical')

    # Tick labels for x axis
    plt.xticks(varimp.index, varimp['variable'], rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title(run_name + ' Variable Importances')

    # ax = plt.axes()
    ax.xaxis.grid(False)  # remove just xaxis grid

    plt.savefig(run_name + '_importance-graph.png')
#     plt.close()
    # self.impgraph = plt
    varimp = varimp