from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd


def train_reg(algorithm, estimator, train_features, train_target, test_features, test_target, fit_params=None, n=5):
    """
    Function to train the model n times and collect basic statistics about results.
    :param self:
    :param n: number of replicates
    :return:
    """

    print("Starting model training with {} replicates.\n".format(n), end=' ', flush=True)

    # empty arrays for storing replicated data
    r2 = np.empty(n)
    mse = np.empty(n)
    rmse = np.empty(n)
    
    pva = pd.DataFrame([], columns=['actual', 'predicted'])

    for i in range(n):
        
        if algorithm == "nn":
            estimator.fit(train_features, train_target, **fit_params)

        else:
            estimator.fit(train_features, train_target)

        predictions = estimator.predict(test_features)
        
        #Dataframe for replicate_model
        pva['actual'] = test_target
        pva['predicted'] = predictions

        r2[i] = r2_score(pva['actual'], pva['predicted'])
        mse[i] = mean_squared_error(pva['actual'], pva['predicted'])
        rmse[i] = np.sqrt(mean_squared_error(pva['actual'], pva['predicted']))
        # store as enumerated column for multipredict
        pva['predicted' + str(i)] = predictions
    
    pva_scaled = pva
    # Holding variables for scaled data
    scaled_r2 = np.empty(n+1)
    scaled_mse = np.empty(n+1)
    scaled_rmse = np.empty(n+1)
    
    data_max = max(pva_scaled.max())  # Find abs min/max of predicted data
    data_min = min(pva_scaled.min())

    # Logic to scale the predicted data, using min/max scaling
    pva_scaled = (pva_scaled - data_min) / (data_max - data_min)
    predicted_columns = pva_scaled.columns.difference(['actual'])
    #pva_scaled.to_csv("test_pva_scaled.csv")
    # Calculate r2, rmse, mse or for each pva columns
    for i, predicted_column in enumerate(predicted_columns):
        scaled_r2[i] = r2_score(pva_scaled['actual'], pva_scaled[predicted_column])
        scaled_mse[i] = mean_squared_error(pva_scaled['actual'], pva_scaled[predicted_column])
        scaled_rmse[i] = np.sqrt(scaled_mse[i])
    
    #print(np.isnan(scaled_r2).any()) 
    #Gather MSE, RMSE, and STD for each molecule in the predictions and scaled_predictions csv files
    def __gather_column_stats__(pva_df):
        pva_df['pred_avg'] = pva_df[predicted_columns].mean(axis=1)
        pva_df['pred_std'] = pva_df[predicted_columns].std(axis=1)
        pva_df['pred_average_error'] = abs(pva_df['actual'] - pva_df['pred_avg'])
        return pva_df

    pva_scaled = __gather_column_stats__(pva_scaled)
    pva = __gather_column_stats__(pva)

    stats = {
        'r2_raw': r2,
        'r2_avg': r2.mean(),
        'r2_std': r2.std(),
        'mse_raw': mse,
        'mse_avg': mse.mean(),
        'mse_std': mse.std(),
        'rmse_raw': rmse,
        'rmse_avg': rmse.mean(),
        'rmse_std': rmse.std(),
    }

    scaled_stats = {
        'r2_raw_scaled': scaled_r2,
        'r2_avg_scaled': scaled_r2.mean(),
        'r2_std_scaled': scaled_r2.std(),
        'mse_raw_scaled': scaled_mse,
        'mse_avg_scaled': scaled_mse.mean(),
        'mse_std_scaled': scaled_mse.std(),
        'rmse_raw_scaled': scaled_rmse,
        'rmse_avg_scaled': scaled_rmse.mean(),
        'rmse_std_scaled': scaled_rmse.std(),
    }

    predictions = pva
    predictions_stats = stats
            
    scaled_predictions = pva_scaled
    scaled_predictions_stats = scaled_stats
                    
    return predictions, predictions_stats, scaled_predictions, scaled_predictions_stats
