from pathlib import Path

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
    varimp.to_csv("importances.csv")
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

    plt.savefig(run_name + '_importance-graph.png')
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


def main():
    si_aerogel_file = str(Path(__file__).parent.parent / "files/si_aerogels/si_aerogel_AI_machine_readable.csv")
    state = None
    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Thermal Conductivity (W/mK)',
                    'Crystalline Phase', 'Nanoparticle Size (nm)', 'Average Pore Size (nm)']
    y_column = 'Surface Area (m2/g)'

    # Data processing
    all_data = read_csv(si_aerogel_file)
    all_data = all_data.drop(drop_columns, axis=1)  # Remove other final gel properties
    all_data = all_data.dropna(how='any', subset=[y_column])  # Drop rows that dont specify surface area
    all_data = all_data.fillna(0)  # Replace blank spaces with 0
    all_data = get_dummies(all_data)  # Replace words with dummy numbers
    y_column_max, y_column_min = all_data[y_column].max(), all_data[y_column].min()
    all_data[y_column] = (all_data[y_column] - y_column_min) / (y_column_max - y_column_min)  # Scaled data
    all_data = shuffle(all_data, random_state=state)

    y = all_data[y_column]
    x = all_data.drop([y_column], axis=1)

    feature_list = x.columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=state)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.875, random_state=state)

    pva = DataFrame({"actual": y_test})

    feature_importances = DataFrame()
    predicted_columns = []
    for i in tqdm(range(100)):
        reg = RandomForestRegressor()
        reg.fit(x_train, y_train)
        y_predicted = reg.predict(x_test)
        predicted_column = f"predicted {i}"
        pva[predicted_column] = y_predicted
        predicted_columns.append(predicted_column)
        feature_importances[predicted_column] = reg.feature_importances_

    pva['pred_avg'] = pva[predicted_columns].mean(axis=1)
    pva['pred_std'] = pva[predicted_columns].std(axis=1)
    pva_graph(pva, "scaled_preliminary_results_rf_100_runs")

    pva = (y_column_max - y_column_min) * pva + y_column_min
    pva_graph(pva, "unscaled_preliminary_results_rf_100_runs")
    feature_importances = feature_importances.mean(axis=1).to_numpy()
    impgraph(feature_importances, feature_list, "preliminary_results_rf_100_runs")


if __name__ == "__main__":
    main()
