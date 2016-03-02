#! bin/var/env python2.7
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from xgboost.sklearn import XGBClassifier

__author__ = 'sebastien.genty'
__version__ = '1.0'


def xgboostinitial_predictor(train_path, test_path, eval_path):
    # Loading the data
    print 'Loading the data...'
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    eval_df = pd.read_csv(eval_path, index_col=0)
    target = train['target'].copy()
    train.drop('target', axis=1, inplace=True)

    # Training model
    print 'Model training begins...'
    # xgtrain = xgb.DMatrix(train.values, target.values, missing=np.nan)
    # xgboost_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'eval_metric': 'logloss', 'eta': 0.01,
    #                   'subsample': 0.5, 'colsample_bytree': 0.5, 'max_depth': 10, 'silent': 0}
    #
    # xgb_model = xgb.train(xgboost_params, xgtrain, learning_rates=0.3)

    xgb_model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='binary:logistic',
                              subsample=0.5, colsample_bytree=0.5, seed=0)
    xgb_model.fit(train.as_matrix(), target.tolist())

    # Running the model
    print 'Making predictions....'
    # xgtest = xgb.DMatrix(test.values)
    # xgeval = xgb.DMatrix(eval_df)

    test_preds = xgb_model.predict_proba(test.as_matrix())
    eval_preds = xgb_model.predict_proba(eval_df.as_matrix())

    print 'Cleaning predictions to match expected format....'
    test_output = pd.DataFrame(test_preds, index=test.index)
    print test_output.columns
    test_output = test_output[1]
    test_output.columns = ['PredictedProb']

    eval_output = pd.DataFrame(eval_preds, index=eval_df.index)
    eval_output = eval_output[1]
    eval_output.columns = ['PredictedProb']

    return test_output, eval_output


def main():
    # Printing start time to know how long this thing's been going on for
    print datetime.now().time().isoformat()
    pd.options.mode.chained_assignment = None

    # Running the model
    train_path = 'training_uncat_df.csv'
    test_path = 'training untagged.csv'
    eval_path = 'test_df.csv'
    test_preds, eval_preds = xgboostinitial_predictor(train_path, test_path, eval_path)

    # Saving the data
    print 'Saving predictions...'
    test_preds.to_csv('xgb init predictions.csv')
    eval_preds.to_csv('xgb mean preditctions.csv')

if __name__ == '__main__':
    main()
