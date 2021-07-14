from shutil import rmtree
from pathlib import Path
from os import getcwd

from pandas import DataFrame
from numpy import array, sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras_tuner import Hyperband

from machine_learning import graph


class HyperTuneKeras:

    def __init__(self, train_features, train_target, seed=None, validation_percent=0.1):
        self.seed = seed

        tp = 1 - validation_percent
        self.train_features, self.val_features, self.train_target, self.val_target = train_test_split(train_features,
                                                                                                      train_target,
                                                                                                      train_size=tp,
                                                                                                      random_state=seed)
        self.tuner = None
        self.estimator = None
        self.params = None

    @staticmethod
    def one_hidden_layer(hp):
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
                hp.Choice("learning_rate", values=[2e-3, 1e-3, 9e-4]),
            ),
            loss="mse",
            metrics=["mse", "mae"],
        )
        return model

    def tune(self):
        self.tuner = Hyperband(
            self.one_hidden_layer,
            objective="val_loss",
            max_epochs=20,
            hyperband_iterations=5,
            project_name="untitled_project"
        )

        self.tuner.search(self.train_features, self.train_target,
                          validation_data=(self.val_features, self.val_target),
                          callbacks=[TensorBoard('untitled_project')])
        self.params = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        self.estimator = self.tuner.hypermodel.build(self.params)
        # tune_score = None  # TODO get tune score

        rmtree("untitled_project")

        return self.estimator, self.params

    def plot_overfit(self, run_name):
        history = self.estimator.fit(self.train_features, self.train_target, epochs=20,
                                     validation_data=(self.val_features, self.val_target))
        plt.plot(history.history['mae'], '-*')
        plt.plot(history.history['val_mse'], '--o')
        plt.title('MSE: Train and Validation', fontsize=14, fontweight='bold')
        plt.ylabel('MSE', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12, fontweight='bold')
        plt.legend(["Train", "Validation"], loc="upper left")
        output_file = Path(getcwd()) / f"{run_name}_overfit.png"
        plt.savefig(output_file)

    def plot_val_pva(self, run_name, target_scaler):

        pva = None
        r2 = []

        tms = MinMaxScaler()  # target minmax scaler

        # Return train target to original scaling
        scaled_train_target = target_scaler.inverse_transform(self.train_target.reshape(-1, 1)).reshape(-1, )
        # MinMax Scale train target
        scaled_train_target = tms.fit_transform(scaled_train_target.reshape(-1, 1)).reshape(-1, )

        for i in range(5):

            model = self.tuner.hypermodel.build(self.params)
            model.fit(self.train_features, self.train_target, epochs=20)
            predictions = model.predict(self.val_features)

            # Return predictions to original scaling
            scaled_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, )
            # MinMax Scale predictions
            scaled_predictions = tms.transform(scaled_predictions.reshape(-1, 1)).reshape(-1, )

            r2.append(r2_score(scaled_train_target, scaled_predictions))
            if i == 0:
                pva = DataFrame(predictions, columns=[f"pred_{i}"])
            else:
                pva[f"pred_{i}"] = scaled_predictions

        pva["pred_std"] = pva.std(axis=1)
        pva["pred_avg"] = pva.mean(axis=1)
        pva["actual"] = scaled_train_target

        r2 = array(r2)
        r2_std = r2.std()
        r2_avg = r2.mean()

        prediction_stats = dict(
            r2_avg=r2_avg,
            r2_std=r2_std,
            mse_avg=mean_squared_error(pva['actual'], pva['pred_avg']),
            rmse_avg=sqrt(mean_squared_error(pva['actual'], pva['pred_avg'])),
            pred_std=pva["pred_std"],
            pred_avg=pva["pred_avg"],
        )

        graph.pva_graph(prediction_stats, pva, run_name, scaled=False)
