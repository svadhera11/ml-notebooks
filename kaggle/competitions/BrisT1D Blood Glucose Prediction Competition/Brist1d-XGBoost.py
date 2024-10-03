import os
import time
import pickle
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor



def read_data():
    
    df_train = pd.read_csv('brist1d/train.csv', low_memory=False)
    df_test = pd.read_csv('brist1d/test.csv', low_memory=False)
    with open('brist1d/activities.txt', 'r') as f:
            activity_list = f.read()
            activity_list = activity_list.replace('\n', ',').split(',')[:-1]
    activity_list.append('CoreTraining') # manually added two activities that were not in activity_list.
    activity_list.append('Cycling')
    
    df_x = df_train.drop(['bg+1:00'], axis=1)
    df_y = df_train[['bg+1:00']]
    df_x_test = df_test
    
    key_parameters = list(set([df_x.keys()[i].split('-')[0] for i in range(df_x.shape[1])]))
    cols = df_x.columns.str.split('-', expand=True)
    df_x.columns = pd.MultiIndex.from_tuples(cols)
    df_x_test.columns = pd.MultiIndex.from_tuples(cols)
    
    print("IMPUTING AND SCALING...")
    
    df_x_train, df_x_val, df_y_train, df_y_val = train_test_split(df_x, df_y, test_size=0.3, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    f = df_x_train[['carbs','cals','steps','hr','bg','insulin']]
    fp = df_x_test[['carbs','cals','steps','hr','bg','insulin']]
    fq = df_x_val[['carbs','cals','steps','hr','bg','insulin']]
    
    g = df_x_train[['activity']]
    gp = df_x_test[['activity']]
    gq = df_x_val[['activity']]
    
    f = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(f)), columns = f.columns).reset_index(drop=True)
    fp = pd.DataFrame(scaler.transform(imputer.transform(fp)), columns = fp.columns).reset_index(drop=True)
    fq = pd.DataFrame(scaler.transform(imputer.transform(fq)), columns = fq.columns).reset_index(drop=True)
    
    g = g.map(lambda x: 0 if x is np.nan else activity_list.index(x)+1).reset_index(drop=True)
    gp = gp.map(lambda x: 0 if x is np.nan else activity_list.index(x)+1).reset_index(drop=True)
    gq = gq.map(lambda x: 0 if x is np.nan else activity_list.index(x)+1).reset_index(drop=True)
    
    df_x_train = pd.concat([f, g], axis=1)
    df_x_test = pd.concat([fp, gp], axis=1)
    df_x_val = pd.concat([fq, gq], axis=1)
    
    print("DONE!")

    return df_x_train, df_x_val, df_y_train, df_y_val, df_x_test



def load_or_process_data():
    
    if os.path.exists('processed_input.pkl'):
        print("Loading processed data from 'processed_input.pkl'...")
        with open('processed_input.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Data Loading Complete.")
        return data['x_train'], data['x_val'], data['y_train'], data['y_val'], data['x_test']
    else:
        df_x_train, df_x_val, df_y_train, df_y_val, df_x_test = read_data()
        with open('processed_input.pkl', 'wb') as f:
            pickle.dump({
                'x_train': df_x_train,
                'x_val': df_x_val,
                'y_train': df_y_train,
                'y_val': df_y_val,
                'x_test': df_x_test
            }, f)
        print("Data processing complete and saved to 'processed_input.pkl'.")
        return df_x_train, df_x_val, df_y_train, df_y_val, df_x_test

def xgboost_model(x_train, x_val, y_train, y_val, x_test):
    
    XGModel = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.1, n_jobs=-1,
                           random_state=42)
    XGModel.fit(x_train, y_train)
    

    y_train_pred_XGModel = XGModel.predict(x_train)
    y_val_pred_XGModel = XGModel.predict(x_val)
    y_test_pred_XGModel = XGModel.predict(x_test)
    

    train_mse_XGModel = mean_squared_error(y_train, y_train_pred_XGModel)
    val_mse_XGModel = mean_squared_error(y_val, y_val_pred_XGModel)
    print("---------------------------------------------")
    print(f"XGModel Training MSE: {train_mse_XGModel:.3f}")
    print(f"XGModel Validation MSE: {val_mse_XGModel:.3f}")
    print("---------------------------------------------")
    

    XGModel_dict = {
        'XGModel': XGModel,
        'y_train_pred_XGModel': y_train_pred_XGModel,
        'y_val_pred_XGModel': y_val_pred_XGModel,
        'y_test_pred_XGModel': y_test_pred_XGModel,
        'train_mse_XGModel': train_mse_XGModel,
        'val_mse_XGModel': val_mse_XGModel
    }
    

    with open('XGModel.pkl', 'wb') as f:
        pickle.dump(XGModel_dict, f)
    
    print("Model and variables have been saved successfully.")
    
    return None


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
    study.optimize(objective, n_trials=100)  # Run 100 trials
    
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
    
    OptunaXGModel_best_data_to_save = {
        'OptunaXGModel_best': OptunaXGModel_best,
        'best_params': best_params,
        'y_train_pred_OptunaXGModel_best': y_train_pred_OptunaXGModel_best,
        'y_val_pred_OptunaXGModel_best': y_val_pred_OptunaXGModel_best,
        'y_test_pred_OptunaXGModel_best': y_test_pred_OptunaXGModel_best,
        'train_mse_OptunaXGModel_best': train_mse_OptunaXGModel_best,
        'val_mse_OptunaXGModel_best': val_mse_OptunaXGModel_best
    }
    
    with open('OptunaXGModel_best.pkl', 'wb') as f:
        pickle.dump(OptunaXGModel_best_data_to_save, f)
    
    print("Model, hyperparameters, predictions, and metrics have been saved successfully.")

    return None

#################################################################################################
load_tick = time.time()
df_x_train, df_x_val, df_y_train, df_y_val, df_x_test = load_or_process_data()
x_train = df_x_train.to_numpy()
x_val = df_x_val.to_numpy()
x_test = df_x_test.to_numpy()
y_train = df_y_train.to_numpy().reshape(-1)
y_val = df_y_val.to_numpy().reshape(-1)
load_tock = time.time()
print(f"Loading Data took {(load_tock - load_tick):.5f} seconds")
xg_tick = time.time()
xgboost_model(x_train, x_val, y_train, y_val, x_test)
xg_tock = time.time()
print(f"XGBoost Model took {(xg_tock - xg_tick):.5f} seconds")
opt_xg_tick = time.time()
optuna_xgboost_model(x_train, x_val, y_train, y_val, x_test)
opt_xg_tock = time.time()
print(f"Optuna + XGBoost Model took {(opt_xg_tock - opt_xg_tick):.5f} seconds")
##################################################################################################
