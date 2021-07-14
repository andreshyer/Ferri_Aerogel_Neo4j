from pathlib import Path
from random import randint
from shutil import rmtree

from numpy import nan

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras_tuner import Hyperband, RandomSearch
from tensorflow import keras

from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel, DataFrame
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

from machine_learning import DataSplitter, Scaler, graph
from machine_learning.featurization import featurize_si_aerogels


# gather data
dataset = r"machine_learning_si_aerogels.xlsx.csv"
folder = "si_aerogels"
file_path = "files/si_aerogels/machine_learning_si_aerogels.xlsx/"

data_path = str(Path(__file__).parent.parent / file_path)
data = read_excel(data_path)

y_columns = ['Surface Area m2/g']
drop_columns = ['Title', 'Porosity', 'Porosity %', 'Pore Volume cm3/g', 'Average Pore Diameter nm',
                'Bulk Density g/cm3', 'Young Modulus MPa', 'Crystalline Phase',
                'Average Pore Size nm', 'Thermal Conductivity W/mK', 'Gelation Time mins']


data = featurize_si_aerogels(df=data, str_method="one_hot_encode", num_method="smart_values",
                             y_columns=y_columns, drop_columns=drop_columns, remove_xerogels=True)


splitter = DataSplitter(df=data, y_columns=y_columns, train_percent=70,
                        test_percent=20, val_percent=10, state=None)
x_test, x_train, x_val, y_test, y_train, y_val, feature_list = splitter.split_data()

features_scaler = StandardScaler()
target_scaler = StandardScaler()

# Scale training features/target
x_train = features_scaler.fit_transform(x_train)
x_test = features_scaler.transform(x_test)
x_val = features_scaler.transform(x_val)

# Scale testing features/target
y_train = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1, )
y_test = target_scaler.transform(y_test.to_numpy().reshape(-1, 1)).reshape(-1, )
y_val = target_scaler.transform(y_val.to_numpy().reshape(-1, 1)).reshape(-1, )

# model = Sequential()
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1, name='output_layer'))
# model.compile(loss='mse', optimizer='adam', metrics=["mse", "mae"])
#
# history = model.fit(x_train, y_train, batch_size=5, epochs=10, validation_data=(x_val, y_val))

#
# plt.plot(history.history['mae'], '-*')
# plt.plot(history.history['val_mae'], '--o')
# plt.title('MAE: Train and Validation', fontsize=14, fontweight='bold')
# plt.ylabel('MAE', fontsize=12, fontweight='bold')
# plt.xlabel('Epochs', fontsize=12, fontweight='bold')
# plt.legend(["Train", "Validation"], loc="upper left")
# plt.savefig("Overfit.png")


def no_dropout_model(hp):
    model = Sequential()
    model.add(
        Dense(
            kernel_initializer='normal',
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    # model.add(
    #     Dense(
    #         units=hp.Int("units", min_value=8, max_value=512, step=8),
    #         activation='relu',
    #     )
    # )
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", values=[1e-3, 1e-5, 1e-7, 1e-9]),
        ),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


def dropout_model(hp):
    model = Sequential()
    model.add(
        Dense(
            kernel_initializer='normal',
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    model.add(
        Dense(
            units=hp.Int("units", min_value=8, max_value=512, step=8),
            activation='relu',
        )
    )
    model.add(
        Dropout(
            hp.Float(
                'dropout',
                min_value=0.0,
                max_value=0.1,
                default=0.005,
                step=0.01)
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(
            hp.Choice("learning_rate", values=[1e-3, 1e-5, 1e-7, 1e-9]),
        ),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse')
    return model


# tuner = RandomSearch(
#     no_dropout_model,
#     objective="val_loss",
#     max_trials=1
# )


tuner = Hyperband(
    no_dropout_model,
    objective="val_loss",
    max_epochs=100,
    hyperband_iterations=1,
    project_name="untitled_project"
)

# tuner.search_space_summary()
tuner.search(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val),
             callbacks=[tf.keras.callbacks.TensorBoard('untitled_project')])

# model: Sequential() = tuner.get_best_models(num_models=5)[0]

# tuner.results_summary()


# pred_results = model.predict(x_train)

# pva = DataFrame(y_test, columns=['pred_actual'])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
rmtree("untitled_project")

model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
plt.plot(history.history['mae'], '-*')
plt.plot(history.history['val_mae'], '--o')
plt.title('MAE: Train and Validation', fontsize=14, fontweight='bold')
plt.ylabel('MAE', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("Overfit.png")

pva = None
r2 = []
for i in range(5):
    model = tuner.hypermodel.build(best_hps)
    model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
    predictions = model.predict(x_test)
    r2.append(r2_score(y_test, predictions))
    if i == 0:
        pva = DataFrame(predictions, columns=[f"pred_{i}"])
    else:
        pva[f"pred_{i}"] = predictions

pva["pred_std"] = pva.std(axis=1)
pva["pred_avg"] = pva.mean(axis=1)
pva["actual"] = y_test

r2 = np.array(r2)
r2_std = r2.std()
r2_avg = r2.mean()

prediction_stats = dict(
    r2_avg=r2_avg,
    r2_std=r2_std,
    mse_avg=mean_squared_error(pva['actual'], pva['pred_avg']),
    rmse_avg=np.sqrt(mean_squared_error(pva['actual'], pva['pred_avg'])),
    pred_std=pva["pred_std"],
    pred_avg=pva["pred_avg"],
)

graph.pva_graph(prediction_stats, pva, "dev", scaled=False)
# print(history)
