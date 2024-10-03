import os
import time
import pickle
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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



# Search for best hyperparameters for XGBoost using Optuna.
def optuna_xgboost_model(x_train, x_val, y_train, y_val, x_test):
    
    def objective(trial, x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, x_test=x_test):
        ''' Objective Function for Optuna '''
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma = trial.suggest_float('gamma', 0, 5)

        OptunaXGModel = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree = colsample_bytree, 
            gamma = gamma,
            n_jobs=-1,
            random_state=42
        )

        OptunaXGModel.fit(x_train, y_train)

        y_val_pred_OptunaXGModel = OptunaXGModel.predict(x_val)

        val_mse_OptunaXGModel = mean_squared_error(y_val, y_val_pred_OptunaXGModel)

        return val_mse_OptunaXGModel
    
    study = optuna.create_study(direction='minimize')  # Minimizing validation MSE
    study.optimize(objective, n_trials=500)  # Run 100 trials
    
    best_params = study.best_params
    
    OptunaXGModel_best = XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma = best_params['gamma'],
        n_jobs=-1,
        random_state=42
    )
    
    OptunaXGModel_best.fit(x_train, y_train)
    
    y_train_pred_OptunaXGModel_best = OptunaXGModel_best.predict(x_train)
    y_val_pred_OptunaXGModel_best = OptunaXGModel_best.predict(x_val)
    y_test_pred_OptunaXGModel_best = OptunaXGModel_best.predict(x_test)
    
    train_mse_OptunaXGModel_best = mean_squared_error(y_train, y_train_pred_OptunaXGModel_best)
    val_mse_OptunaXGModel_best = mean_squared_error(y_val, y_val_pred_OptunaXGModel_best)

    print("---------------------------------------------")
    print(f"Optuna-Tuned XGModel Training MSE: {train_mse_OptunaXGModel_best:.3f}")
    print(f"Optuna-Tuned XGModel Validation MSE: {val_mse_OptunaXGModel_best:.3f}")
    print("---------------------------------------------")
    
    OptunaXGMetaModel_best_data_to_save = {
        'OptunaXGModel_best': OptunaXGModel_best,
        'best_params': best_params,
        'y_train_pred_OptunaXGModel_best': y_train_pred_OptunaXGModel_best,
        'y_val_pred_OptunaXGModel_best': y_val_pred_OptunaXGModel_best,
        'y_test_pred_OptunaXGModel_best': y_test_pred_OptunaXGModel_best,
        'train_mse_OptunaXGModel_best': train_mse_OptunaXGModel_best,
        'val_mse_OptunaXGModel_best': val_mse_OptunaXGModel_best
    }
    
    with open('OptunaXGMetaModel_best.pkl', 'wb') as f:
        pickle.dump(OptunaXGMetaModel_best_data_to_save, f)
    
    print("Model, hyperparameters, predictions, and metrics have been saved successfully.")

    return None

####################################################################################################
read_tick = time.time()
x_train, x_val, y_train, y_val, x_test = import_data()
read_tock = time.time()
print(f'reading in data took {read_tock - read_tick:.5f} seconds')
optuna_tick = time.time()
optuna_xgboost_model(x_train, x_val, y_train, y_val, x_test)
optuna_tock = time.time()
print(f'optuna optimization took {optuna_tock - optuna_tick:.5f} seconds')
####################################################################################################
