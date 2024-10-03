# importing necessary libraries
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
from sklearn.ensemble import RandomForestRegressor


# importing the requisite data and pre-processing it.

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


# defining the models, fitting them, and returning outputs.

# Random Forest.
def random_forest_model(x_train, x_val, y_train, y_val, x_test):
    
    RFModel = RandomForestRegressor(n_estimators=1_000, max_depth=1_00, max_features='sqrt', n_jobs=-1,
                                  random_state=42)
    RFModel.fit(x_train, y_train) # fit RFModel
    # predictions on RFModel.
    y_train_pred_RFModel = RFModel.predict(x_train)
    y_val_pred_RFModel = RFModel.predict(x_val)
    y_test_pred_RFModel = RFModel.predict(x_test)
    # compute MSE, the metric we're interested in.
    train_mse_RFModel = mean_squared_error(RFModel.predict(x_train), y_train)
    val_mse_RFModel = mean_squared_error(RFModel.predict(x_val), y_val)
    print("---------------------------------------------")
    print(f"RFModel Training MSE: {train_mse_RFModel:.3f}")
    print(f"RFModel Validation MSE: {val_mse_RFModel:.3f}")
    print("---------------------------------------------")
    # save all this in a file.
    
    RFModel_dict = {
        'RFModel': RFModel,
        'y_train_pred_RFModel': y_train_pred_RFModel,
        'y_val_pred_RFModel': y_val_pred_RFModel,
        'y_test_pred_RFModel': y_test_pred_RFModel,
        'train_mse_RFModel': train_mse_RFModel,
        'val_mse_RFModel': val_mse_RFModel
    }
    
    # Save the dictionary to a file
    with open('RFModel.pkl', 'wb') as f:
        pickle.dump(RFModel_dict, f)
    
    print("Model and variables have been saved successfully.")
    
    return None

# Search for best hyperparameters for Random Forests using Optuna.
def optuna_random_forest_model(x_train, x_val, y_train, y_val, x_test):
    
    def objective(trial, x_train = x_train, x_val = x_val, y_train = y_train, y_val = y_val, x_test = x_test):
        ''' Objective Function for Optuna '''
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 10, 100)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    
        # Create the Random Forest model with suggested hyperparameters
        OptunaRFModel = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )
    
        # Train the model
        OptunaRFModel.fit(x_train, y_train)
    
        # Predictions on validation set
        y_val_pred_OptunaRFModel = OptunaRFModel.predict(x_val)
    
        # Calculate validation MSE
        val_mse_OptunaRFModel = mean_squared_error(y_val, y_val_pred_OptunaRFModel)
    
        return val_mse_OptunaRFModel
    
    # Create a study and optimize
    study = optuna.create_study(direction='minimize')  # Minimizing validation MSE
    study.optimize(objective, n_trials=100)  # Run 100 trials
    
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
    
    OptunaRFModel_best_data_to_save = {
        'OptunaRFModel_best': OptunaRFModel_best,
        'best_params': best_params,
        'y_train_pred_OptunaRFModel_best': y_train_pred_OptunaRFModel_best,
        'y_val_pred_OptunaRFModel_best': y_val_pred_OptunaRFModel_best,
        'y_test_pred_OptunaRFModel_best': y_test_pred_OptunaRFModel_best,
        'train_mse_OptunaRFModel_best': train_mse_OptunaRFModel_best,
        'val_mse_OptunaRFModel_best': val_mse_OptunaRFModel_best
    }
    
    with open('OptunaRFModel_best.pkl', 'wb') as f:
        pickle.dump(OptunaRFModel_best_data_to_save, f)
    
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
print(f"Loading Data took {(load_tock-load_tick):.5f} seconds")
rf_tick = time.time()
random_forest_model(x_train, x_val, y_train, y_val, x_test)
rf_tock = time.time()
print(f"Random Forest Model took {(rf_tock-rf_tick):.5f} seconds")
opt_rf_tick = time.time()
optuna_random_forest_model(x_train, x_val, y_train, y_val, x_test)
opt_rf_tock = time.time()
print(f"Optuna + Random Forest Model took {(opt_rf_tock - opt_rf_tick):.5f} seconds")
##################################################################################################
