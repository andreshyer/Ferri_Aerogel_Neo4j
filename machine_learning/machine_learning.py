from pathlib import Path
from datetime import datetime
from os import mkdir, path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas import read_csv, get_dummies, DataFrame, set_option
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def impgraph(feature_importances, feature_list: list, run_name: str):
    """
    Objective: Make a feature importance graph. I'm limiting this to only rf and gdb since only they have feature
    importance (I might need to double check on that). I'm also limiting this to only rdkit2d since the rest are only 0s
    and 1s
    """

    # Get numerical feature importances

    importances2 = feature_importances  # used later for graph

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(feature_list, list(importances2))]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    feature_list = []
    importances2 = []
    for feature, importance in feature_importances:
        if importance > 0:
            feature_list.append(feature)
            importances2.append(importance)
    importances2 = np.array(importances2)

    # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # prepare importance data for export and graphing
    indicies = (-importances2).argsort()
    varimp = DataFrame([], columns=['variable', 'importance'])
    varimp['variable'] = [feature_list[i] for i in indicies]
    varimp['importance'] = importances2[indicies]
    varimp.to_csv(f"{run_name}/importance.csv")
    # Importance Bar Graph
    plt.rcParams['figure.figsize'] = [15, 20]

    # Set the style
    plt.style.use('bmh')

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

    plt.savefig(run_name + '/importance_graph.png')
    plt.close()


def pva_graph(pva, run_name):
    """
    Make Predicted vs. Actual graph with prediction uncertainty.
    Pass dataframe from multipredict function. Return a graph.
    """

    r2 = r2_score(pva['actual'], pva['pred_avg'])
    mse = mean_squared_error(pva['actual'], pva['pred_avg'])
    rmse = np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg']))

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
    plt.title(run_name + f' Predicted vs. Actual')

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
    plt.close()
    # plt.show()
    # self.pva_graph = plt
    # return plt


# def hyper_tune():
#     bayes_grid = {
#         'n_estimators': Integer(100, 2000),
#         'max_features': Categorical(['auto', 'sqrt']),
#         'max_depth': Integer(1, 30),
#         'min_samples_split': Integer(2, 30),
#         'min_samples_leaf': Integer(2, 30),
#         'bootstrap': Categorical([True, False])
#     }

def replace_words_with_numbers(df: DataFrame):
    keywords = [0]

    for column in df:
        if df[column].dtype == "object":
            keywords.extend(df[column].unique())
    keywords = set(keywords)
    keywords = dict(zip(keywords, range(len(keywords))))
    df = df.replace(keywords)
    return df


def main():
    si_aerogel_file = str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_AI_machine_readable.csv")
    state = None
    # drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
    #                 'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Thermal Conductivity (W/mK)',
    #                 'Crystalline Phase', 'Nanoparticle Size (nm)', 'Average Pore Size (nm)', 'Surface Area (m2/g)']
    drop_columns = []
    y_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                 'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Thermal Conductivity (W/mK)',
                 'Crystalline Phase', 'Nanoparticle Size (nm)', 'Average Pore Size (nm)', 'Surface Area (m2/g)']
    only_columns = []
    model = "random_forest"
    # model = "neural_network"

    # Data processing
    all_data = read_csv(si_aerogel_file)
    all_data = all_data.drop(drop_columns, axis=1)  # Remove other final gel properties
    all_data = all_data.fillna(0)  # Replace blank spaces with 0
    all_data = replace_words_with_numbers(all_data)  # Replace words with dummy numbers
    all_data = shuffle(all_data, random_state=state)
    all_data.to_csv('dev.csv')
    actual = all_data[y_columns]
    all_data = all_data.dropna(how='any', subset=y_columns)  # Drop rows that dont specify surface area
    # for y_column in y_columns:
    #     y_column_max, y_column_min = all_data[y_column].max(), all_data[y_column].min()
    #     all_data[y_column] = (all_data[y_column] - y_column_min) / (y_column_max - y_column_min)  # Scaled data

    y = all_data[y_columns]
    x = all_data.drop(y_columns, axis=1)

    feature_list = x.columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=state)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.875, random_state=state)

    now = datetime.now()
    if model == "random_forest":
        feature_importances = DataFrame()
        run_name = "random_forest" + now.strftime("%d_%m_%Y_%H_%M_%S")
    else:
        feature_importances = None
        run_name = "neural_network" + now.strftime("%d_%m_%Y_%H_%M_%S")

    if not path.exists('output'):
        mkdir('output')
    output_dir = f'output/{run_name}'
    mkdir(output_dir)

    predicted_columns = []
    error_pvas = []
    for i in tqdm(range(5)):
        if model == "random_forest":
            reg = RandomForestRegressor()
        else:
            reg = MLPRegressor()
        reg.fit(x_train, y_train)
        y_predicted = reg.predict(x_test)
        predicted_df = DataFrame(y_predicted, columns=actual.columns.tolist())
        error_pva = (actual - predicted_df)**2
        error_pvas.append(error_pva)
        # pva[predicted_column] = y_predicted
        # predicted_columns.append(predicted_column)
        # if isinstance(feature_importances, DataFrame):
        #     feature_importances[predicted_column] = reg.feature_importances_

    rmse_pva = DataFrame()
    for error_pva in error_pvas:
        rmse_pva = rmse_pva + error_pva
    print(rmse_pva)

    # pva['pred_avg'] = pva[predicted_columns].mean(axis=1)
    # pva['pred_std'] = pva[predicted_columns].std(axis=1)
    # pva_graph(pva, f"{output_dir}/scaled")
    #
    # pva = (y_column_max - y_column_min) * pva + y_column_min
    # pva_graph(pva, f"{output_dir}/unscaled")
    #
    # if isinstance(feature_importances, DataFrame):
    #     feature_importances = feature_importances.mean(axis=1).to_numpy()
    #     impgraph(feature_importances, feature_list, output_dir)


if __name__ == "__main__":
    main()
