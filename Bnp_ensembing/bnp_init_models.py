#! bin/var/env python2.7
import numpy as np
import pandas as pd
import h2o
import xgboost as xgb

from datetime import datetime
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from xgboost.sklearn import XGBClassifier

__author__ = 'sebastien.genty'
__version__ = '1.0'


def col_index_generator(df_path):
    # Loading the df
    df = pd.read_csv(df_path, index_col=0)

    # Getting what we need and returning it
    index = df.index
    col_list = df.columns.tolist()

    return index, col_list


def h20initial_predictor(train_path, test_path, eval_path):
    # Getting index and column names
    index, pred_names = col_index_generator(test_path)

    # Loading H2O, getting stuff ready for model
    print 'Connecting to  h2o node...'
    h2o.init(ip='10.1.5.154', port=54321)

    print 'Formatting data for h2o...'
    train_h2o = h2o.upload_file(train_path, header=1, destination_frame='training_data_with_target')
    test_h2o = h2o.upload_file(test_path, header=1, destination_frame='test_data_without_target')
    eval_h20 = h2o.upload_file(eval_path, header=1, destination_frame='eval_data_without_target')

    splits = train_h2o.split_frame(ratios=[0.75], seed=1234)

    print 'Training h2o deep learning module...'
    nn_model = H2ODeepLearningEstimator(distribution='bernoulli', hidden=[200, 200, 200], epochs=50,
                                        activation='RectifierWithDropout', overwrite_with_best_model=True, rho=0.99,
                                        input_dropout_ratio=0.2, epsilon=1e-10, l1=1e-4, l2=1e-4,
                                        stopping_tolerance=1e-10, stopping_metric='logloss')

    nn_model.train(x=pred_names, y='target', training_frame=splits[0], ignored_columns='ID',
                   validation_frame=splits[1])

    # Running the model
    print 'Making predictions....'
    train_preds = nn_model.predict(test_h2o)
    eval_preds = nn_model.predict(eval_h20)

    print 'Cleaning predictions to match expected format....'
    train_output = h2o.as_list(train_preds)['YES']
    train_output.index = index
    train_output.columns = ['PredictedProb']

    eval_index, _ = col_index_generator(eval_path)
    eval_output = h2o.as_list(eval_preds)['YES']
    eval_output.index = eval_index
    eval_output.columns = ['PredictedProb']

    return train_output, eval_output


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
    xgtrain = xgb.DMatrix(train.values, target.values, missing=np.nan)
    xgboost_params = {'objective': 'binary:logistic', 'booster': 'gbtree', 'eval_metric': 'logloss', 'eta': 0.01,
                      'subsample': 1.2, 'colsample_bytree': 0.6, 'max_depth': 12, 'silent': 0, 'min_child_weight': 7,
                      'seed': 0, 'learning_rate': 0.3, 'n_estimators': 70, 'verbose_eval': True}

    xgb_cv = xgb.cv(xgboost_params, xgtrain, num_boost_round=1000, early_stopping_rounds=1e-10, nfold=5,
                    show_progress=True)
    xgb_cv.to_csv('xgv_cv results depth 15.csv')
    xgb_model = xgb.train(xgboost_params, xgtrain, verbose_eval=True, num_boost_round=1000)

    # xgb_model = XGBClassifier(max_depth=10, learning_rate=0.3, n_estimators=25, objective='binary:logistic',
    #                           subsample=1, colsample_bytree=0.2, seed=0, min_child_weight=5)
    # xgb_model.fit(train.as_matrix(), target.tolist(), eval_metric='logloss')

    # Running the model
    print 'Making predictions....'
    xgtest = xgb.DMatrix(test.values)
    xgeval = xgb.DMatrix(eval_df)

    test_preds = xgb_model.predict(xgtest)
    eval_preds = xgb_model.predict(xgeval)

    print 'Cleaning predictions to match expected format....'
    test_output = pd.DataFrame(test_preds, index=test.index)
    print test_output.columns
    # test_output = test_output[1]
    test_output.columns = ['PredictedProb']

    eval_output = pd.DataFrame(eval_preds, index=eval_df.index)
    # eval_output = eval_output[1]
    eval_output.columns = ['PredictedProb']

    return test_output, eval_output


def main():
    # Printing start type
    print datetime.now().time().isoformat()

    # Model function
    train_cat_path = 'training_cat_df.csv'
    train_uncat_path = 'training_uncat_df.csv'
    test_path = 'training untagged.csv'
    eval_path = 'test_df.csv'

    print 'H2o model...'
    # h2o_train_preds, h2o_eval_preds = h20initial_predictor(train_cat_path, test_path, eval_path)
    print 'xgb model...'
    xgb_test_preds, xgb_eval_preds = xgboostinitial_predictor(train_uncat_path, test_path, eval_path)

    print 'Saving predictions...'
    # h2o_train_preds.to_csv('h2o init predictions.csv')
    # h2o_eval_preds.to_csv('h2o mean predictions.csv')

    xgb_test_preds.to_csv('xgb init predictions.csv')
    xgb_eval_preds.to_csv('xgb mean preditctions.csv')

    print 'All is well that ends well. Thank you for using this program!'

if __name__ == '__main__':
    main()