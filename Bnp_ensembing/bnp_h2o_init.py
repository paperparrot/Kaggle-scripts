#! bin/var/env python2.7
import pandas as pd
import h2o

from datetime import datetime
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

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


def main():
    # Printing start type
    print datetime.now().time().isoformat()

    # Model function
    train_path = 'training_cat_df.csv'
    test_path = 'training untagged.csv'
    eval_path = 'test_df.csv'
    train_preds, eval_preds = h20initial_predictor(train_path, test_path, eval_path)

    print 'Saving predictions...'
    train_preds.to_csv('h2o init predictions.csv')
    eval_preds.to_csv('h2o mean predictions.csv')

if __name__ == '__main__':
    main()
