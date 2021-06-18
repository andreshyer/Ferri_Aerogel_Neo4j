import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from xgboost import plot_importance
import shap
import matplotlib


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

    r2 = predictions_stats['r2_avg']
    mse = predictions_stats['mse_avg']
    rmse = predictions_stats['rmse_avg']

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
    if algorithm == "nn":
        Exception("Don't use this feature importance function for " + algorithm)
    else:
        # Get numerical feature importances
        if algorithm == "xgb":
            estimator.get_booster().feature_names = feature_list

            #plot_importance(estimator, max_num_features=20, importance_type='gain')
            #plt.savefig(run_name + '_importance-graph.png')

            importance = estimator.get_booster().get_score(importance_type="weight")
            print(importance)
            tuples = [(k, importance[k]) for k in importance]
            tuples = sorted(tuples, key=lambda x: x[1])[-20:]

        else:
        
            importances2 = estimator.feature_importances_  # used later for graph
            indicies = (-importances2).argsort()
            print(importances2)
            varimp = pd.DataFrame([], columns=['variable', 'importance'])
            varimp['variable'] = [feature_list[i] for i in indicies]
            varimp['importance'] = importances2[indicies]
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
     
            plt.tight_layout()
            # ax = plt.axes()
            ax.xaxis.grid(False)  # remove just xaxis grid
    
            plt.savefig(run_name + '_importance-graph.png')
            # plt.close()
            # self.impgraph = plt


def shap_impgraphs(algorithm, estimator, train_features, feature_list, run_name):
    """
    TODO: FIx force plot. Add docstring
    :param algorithm:
    :param estimator:
    :param train_features:
    :param feature_list:
    :param run_name:
    :return:
    """
    if algorithm in ["rf", "gdb", "xgb"]:
        explainer = shap.TreeExplainer(estimator, feature_names=feature_list)

        shap_values = explainer.shap_values(train_features)
        matplotlib.use('Agg')
        _ = plt.figure()
        shap.summary_plot(shap_values, train_features,feature_names=feature_list, max_display=15)
        plt.tight_layout()
        _.savefig(run_name+"_shap_summary_plot.png")

        g = plt.figure()
        shap.summary_plot(shap_values, train_features,feature_names=feature_list, max_display=15, plot_type='bar')
        plt.tight_layout()
        g.savefig(run_name+"_shap_bar_plot.png")
        
        p = plt.figure()
        shap.force_plot(explainer.expected_value, shap_values, train_features, feature_names=feature_list)
        plt.tight_layout()
        p.savefig(run_name+"_shap_force_plot.png")
        
        #_.tight_layout()
    #plt.close(g)


