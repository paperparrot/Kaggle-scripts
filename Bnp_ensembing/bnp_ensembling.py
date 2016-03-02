#! bin/var/env python2.7

import numpy as np
import pandas as pd
import statsmodels.api as sm

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


def abs_diff(df, col):
    output = df['init target'] - df[col]
    output = output.abs()
    return output


def data_separator(df, value):
    # Selecting right subset of data, removing target and marker variable
    output = df[df['model'] == value].copy()
    output.drop(['model'], axis=1, inplace=True)

    output.fillna(value=0, inplace=True)
    return output


def comparaison_runner(train, xgb, h2o, test):
    # Loading data
    print 'Loading data for comparaison...'
    train_df = pd.read_csv(train, index_col=0)
    xgb_df = pd.read_csv(xgb, index_col=0)
    h2o_df = pd.read_csv(h2o, index_col=0)
    test_df = pd.read_csv(test, index_col=0)

    # Combining results and evaluating best model
    print 'Evaluating estimations...'
    eval_df = pd.concat([train_df['target'], xgb_df, h2o_df], axis=1)
    eval_df.fillna(value=1, inplace=True)

    eval_df.columns = ['init target', 'xgb', 'h2o']
    eval_df['xgb_diff'] = abs_diff(eval_df, 'xgb')
    eval_df['h2o_diff'] = abs_diff(eval_df, 'h2o')

    # Comparing difference, selecting model based on distance
    eval_df.loc[eval_df['xgb_diff'] < eval_df['h2o_diff'], 'model'] = 'xgb'
    eval_df.loc[eval_df['xgb_diff'] >= eval_df['h2o_diff'], 'model'] = 'h2o'
    eval_df.to_csv('eval_results.csv')

    # Getting df's ready for model
    train_df = pd.concat([train_df, eval_df['model']], axis=1, join='inner')
    train_df.drop(['target', 'model'], axis=1, inplace=True)

    # Loading and training model
    print 'Training now begins...'
    model_map = {'xgb': 1, 'h2o': 2}
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(train_df, eval_df['model'].map(model_map))

    print 'Making predictions...'
    model_pred = rfc.predict(test_df)
    model_pred_df = pd.DataFrame(model_pred.tolist(), index=test_df.index)
    model_pred_df.columns = ['model']
    model_pred_df.fillna(value=0, inplace=True)

    print 'Separating Test data for models...'
    test = pd.concat([test_df, model_pred_df], axis=1)
    xgb_test = data_separator(test, 1)
    h2o_test = data_separator(test, 2)

    return xgb_test, h2o_test


def logit_ensembling(eval_path, xgb_mean_path, h2o_mean_path):
    # Loading data, getting stuff ready
    print 'Loading the data for Logit ensembling...'
    eval_results = pd.read_csv(eval_path, index_col=0)
    xgb_result = pd.read_csv(xgb_mean_path, index_col=0,)
    h2o_result = pd.read_csv(h2o_mean_path, index_col=0,  header=None)
    print xgb_result.count()
    print h2o_result.count()
    test_mean = pd.concat([xgb_result, h2o_result], axis=1)
    test_mean.columns = ['xgb', 'h2o']

    logit_preds = eval_results[['xgb', 'h2o']].copy()
    logit_target = eval_results['init target'].copy()

    # Model section
    print 'Logit training...'
    logit_model = RandomForestClassifier()
    logit_model.fit(logit_preds, logit_target)

    results = logit_model.predict_proba(test_mean)
    print results
    ensemble_results = pd.DataFrame(results, index=test_mean.index)
    ensemble_results = ensemble_results[1]
    ensemble_results.columns = ['PredictedProb']

    print ensemble_results.count()

    return ensemble_results


def main():
    # Printing start type
    print datetime.now().time().isoformat()

    train_path = 'training_uncat_df.csv'
    xgb_init = 'xgb init predictions.csv'
    h2o_init = 'h2o init predictions.csv'
    test_path = 'test_df.csv'
    eval_result_path = 'eval_results.csv'
    xgb_mean = 'xgb mean preditctions.csv'
    h2o_mean = 'h2o mean predictions.csv'

    print 'Initial comparasion begins now!...'
    xgb_test, h2o_test = comparaison_runner(train_path, xgb_init, h2o_init, test_path)

    print 'Logit ensembling begins now!...'
    logit_results = logit_ensembling(eval_result_path, xgb_mean, h2o_mean)
    logit_results.to_csv('logit_ensemble_preds.csv')

    xgb_test.to_csv('xgb_test.csv')
    h2o_test.to_csv('h2o_test.csv')

    print 'All is well that ends well. Thank you for using this program!'

if __name__ == '__main__':
    main()
