import os
import time
import pickle
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def submit_best_predictions():
    '''This function will create the submission.csv file. Change the .pkl file name and 'data' dictionary key as required.'''
    
    test_df = pd.read_csv('brist1d/test.csv')
    with open('OptunaRFMetaModel_best.pkl', 'rb') as f:
        data = pickle.load(f)
    submission_df = test_df[['id']]
    submission_df['bg+1:00'] = data['y_test_pred_OptunaRFModel_best']
    submission_df.to_csv('submission-RFMetaModel.csv', header=True, index=None)
    return

submit_best_predictions()
