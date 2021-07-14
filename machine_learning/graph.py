import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from xgboost import plot_importance
import shap
import matplotlib


def pva_graph(predictions_stats, predictions,run_name, scaled=False):
    """
    Make Predicted vs. Actual graph with prediction uncertainty.
    Pass dataframe from multipredict function. Return a graph.
    """
    # Reuse function for scaled data
#     if use_scaled:
#         pva = scaled_predictions
#     else:g
    pva = predictions
    
    if not scaled:
        r2 = predictions_stats['r2_avg']
        mse = predictions_stats['mse_avg']
        rmse = predictions_stats['rmse_avg']
    else:
        r2 = predictions_stats['r2_avg_scaled']
        mse = predictions_stats['mse_avg_scaled']
        rmse = predictions_stats['rmse_avg_scaled']
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
    plt.plot([], [], ' ', label='MSE = %.3f' % mse)
    plt.plot([], [], ' ', label='RMSE = %.3f' % rmse)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.axis([-2,5,-2,5]) #[-2,5,-2,5]
    ax.legend(prop={'size': 16}, facecolor='w', edgecolor='k', shadow=True)

    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent
    if scaled:
        plt.savefig(run_name + '_' + f'_scaled_PVA.png')
    else:
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

            plot_importance(estimator, importance_type='gain')
            #plt.savefig(run_name + '_importance-graph.png')

            importance = estimator.get_booster().get_score(importance_type="weight")
            
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


def abs_shap(df_shap,df, run_name):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist,legend=False)
    
    plt.gcf().set_size_inches(8, len(feature_list) * 0.5 + 1.5)
    plt.tight_layout()
    ax.set_xlabel("mean(|SHAP Value|) (Red = Positive Impact)")
    plt.savefig(run_name + "_summary_bar_plot.png")
    plt.close()


def force_plot(estimator, train_features, run_name, n):
    explainerModel = shap.TreeExplainer(estimator)
    shap_values_Model = explainerModel.shap_values(train_features)
    plt.tight_layout()
    shap.force_plot(explainerModel.expected_value, shap_values_Model[n], train_features.iloc[[n]],show=False,matplotlib=True).savefig(run_name + "_force_plot.png", format = "png",dpi = 150,bbox_inches = 'tight')

    #return(p)
    


def water_fall(estimator, train_features, run_name,n):
    explainerModel = shap.TreeExplainer(estimator)
    shap_values_model = explainerModel.shap_values(train_features)
    shap.plots._waterfall.waterfall_legacy(explainerModel.expected_value, shap_values_model[n])
    plt.savefig(run_name + "_waterfall_plot.png")




def shap_impgraphs(algorithm, estimator, features, feature_list, run_name, predict=False):
    """
    TODO: Add docstring. Try  force plot in Jupyter
    :param algorithm:
    :param estimator:
    :param features:
    :param feature_list:
    :param run_name:
    :return:
    """
    features = pd.DataFrame(features, columns=feature_list)
    if algorithm in ["rf", "gdb", "xgb"]:
        if predict:
            explainer = shap.KernelExplainer(estimator, features)
        else: 
            explainer = shap.TreeExplainer(estimator)

        shap_values = explainer.shap_values(features)
        matplotlib.use('Agg')
        _ = plt.figure()
        shap.summary_plot(shap_values, features, max_display=15)
        plt.tight_layout()
        _.savefig(run_name+"_shap_summary_plot.png")

        g = plt.figure()
        shap.summary_plot(shap_values, features, max_display=15, plot_type='bar')
        plt.tight_layout()
        g.savefig(run_name+"_shap_bar_plot.png")
        plt.close()
        
        abs_shap(shap_values, features, run_name)                
        
        force_plot(estimator, features, run_name, 0)
        
        #water_fall(estimator, features, run_name, 0)

        #_.tight_layout()
    #plt.close(g)


