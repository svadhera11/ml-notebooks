# importing necessary libraries
import os
import time
import pickle
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def import_data():

    if os.path.exists('processed_metamodel_data.pkl'):
        print("Loading data from 'processed_metamodel_data.pkl'...")
        with open('processed_metamodel_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['x_train'], data['x_val'], data['y_train'], data['y_val'], data['x_test']
    else:
        print("File 'processed_metamodel_data.pkl' not found. Loading...")
        with open('OptunaRFModel_best.pkl', 'rb') as f:
            RFModel = pickle.load(f)

        with open('OptunaXGModel_best.pkl', 'rb') as f:
            XGModel = pickle.load(f)

        with open('processed_input.pkl', 'rb') as f:
            processed_input = pickle.load(f)

        train_input_dict = {
            'rfmodel_prediction': RFModel['y_train_pred_OptunaRFModel_best'],
            'xgmodel_prediction': XGModel['y_train_pred_OptunaXGModel_best'],
            'ground_truth': processed_input['y_train'].to_numpy().reshape(-1)
        }
        val_input_dict = {
            'rfmodel_prediction': RFModel['y_val_pred_OptunaRFModel_best'],
            'xgmodel_prediction': XGModel['y_val_pred_OptunaXGModel_best'],
            'ground_truth': processed_input['y_val'].to_numpy().reshape(-1)
        }
        test_input_dict = {
            'rfmodel_prediction': RFModel['y_test_pred_OptunaRFModel_best'],
            'xgmodel_prediction': XGModel['y_test_pred_OptunaXGModel_best']
        }

        train_input_df = pd.DataFrame.from_dict(train_input_dict)
        val_input_df = pd.DataFrame.from_dict(val_input_dict)
        test_input_df = pd.DataFrame.from_dict(test_input_dict)

        train_x = train_input_df[['rfmodel_prediction', 'xgmodel_prediction']].to_numpy()
        val_x = val_input_df[['rfmodel_prediction', 'xgmodel_prediction']].to_numpy()
        train_y = train_input_df['ground_truth'].to_numpy().reshape(-1)
        val_y = val_input_df['ground_truth'].to_numpy().reshape(-1)
        test_x = test_input_df[['rfmodel_prediction', 'xgmodel_prediction']].to_numpy()

        with open('processed_metamodel_data.pkl', 'wb') as f:
            pickle.dump({
                'x_train': train_x,
                'x_val': val_x,
                'y_train':train_y,
                'y_val':val_y,
                'x_test':test_x
            }, f)
        return train_x, val_x, train_y, val_y, test_x

# Search for best hyperparameters for Random Forests using Optuna.
def optuna_random_forest_model(x_train, x_val, y_train, y_val, x_test):
    
    def objective(trial, x_train = x_train, x_val = x_val, y_train = y_train, y_val = y_val, x_test = x_test):
        ''' Objective Function for Optuna '''
        n_estimators = trial.suggest_int('n_estimators', 100, 5000)
        max_depth = trial.suggest_int('max_depth', 10, 500)
        max_features = trial.suggest_categorical('max_features', ['sqrt', None])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    
        OptunaRFModel = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )
    
        OptunaRFModel.fit(x_train, y_train)
    
        y_val_pred_OptunaRFModel = OptunaRFModel.predict(x_val)
    
        val_mse_OptunaRFModel = mean_squared_error(y_val, y_val_pred_OptunaRFModel)
    
        return val_mse_OptunaRFModel
    
    study = optuna.create_study(direction='minimize') 
    study.optimize(objective, n_trials=500)
    
    # Get the best hyperparameters
    best_params = study.best_params
    
    # Train the final model with the best hyperparameters
    OptunaRFModel_best = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the final model on the training set
    OptunaRFModel_best.fit(x_train, y_train)
    
    # Predictions on all sets (train, validation, test)
    y_train_pred_OptunaRFModel_best = OptunaRFModel_best.predict(x_train)
    y_val_pred_OptunaRFModel_best = OptunaRFModel_best.predict(x_val)
    y_test_pred_OptunaRFModel_best = OptunaRFModel_best.predict(x_test)
    
    # Compute MSE on training and validation sets
    train_mse_OptunaRFModel_best = mean_squared_error(y_train, y_train_pred_OptunaRFModel_best)
    val_mse_OptunaRFModel_best = mean_squared_error(y_val, y_val_pred_OptunaRFModel_best)

    print("---------------------------------------------")
    print(f"Optuna-Tuned RFModel Training MSE: {train_mse_OptunaRFModel_best:.3f}")
    print(f"Optuna-Tuned RFModel Validation MSE: {val_mse_OptunaRFModel_best:.3f}")
    print("---------------------------------------------")
    
    OptunaRFMetaModel_best_data_to_save = {
        'OptunaRFModel_best': OptunaRFModel_best,
        'best_params': best_params,
        'y_train_pred_OptunaRFModel_best': y_train_pred_OptunaRFModel_best,
        'y_val_pred_OptunaRFModel_best': y_val_pred_OptunaRFModel_best,
        'y_test_pred_OptunaRFModel_best': y_test_pred_OptunaRFModel_best,
        'train_mse_OptunaRFModel_best': train_mse_OptunaRFModel_best,
        'val_mse_OptunaRFModel_best': val_mse_OptunaRFModel_best
    }
    
    with open('OptunaRFMetaModel_best.pkl', 'wb') as f:
        pickle.dump(OptunaRFMetaModel_best_data_to_save, f)
    
    print("Model, hyperparameters, predictions, and metrics have been saved successfully.")

    return None

##################################################################################################
read_tick = time.time()
x_train, x_val, y_train, y_val, x_test = import_data()
read_tock = time.time()
print(f'reading in data took {read_tock - read_tick:.5f} seconds')
optuna_tick = time.time()
optuna_random_forest_model(x_train, x_val, y_train, y_val, x_test)
optuna_tock = time.time()
print(f'optuna optimization took {optuna_tock - optuna_tick:.5f} seconds')
##################################################################################################
