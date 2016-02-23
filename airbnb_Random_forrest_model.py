import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

__author__ = 'sebastien.genty'
__version__ = '0.5'


def data_cleaner(df):

    # Initialize dataframe
    cleaned_predictors = pd.DataFrame(index=df.index)

    # Transfering columns that need no transformation (already numeric) and source for later use
    cols_to_keep = ['timestamp_first_active', 'age', 'source']
    cleaned_predictors = pd.concat([cleaned_predictors, df[cols_to_keep]], axis=1)

    # Removing outliers in the age data (under 13 and over 95)
    cleaned_predictors.loc[cleaned_predictors.age > 95, 'age'] = np.nan
    cleaned_predictors.loc[cleaned_predictors.age < 13, 'age'] = np.nan

    # Dummyfying all other variables and merging them into the predictors
    dummy_cols = ['gender', 'signup_method', 'signup_flow', 'signup_flow', 'language', 'affiliate_channel',
                  'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

    for i in dummy_cols:
        temp_dummy_df = pd.get_dummies(df[i], prefix=i)
        cleaned_predictors = pd.concat([cleaned_predictors, temp_dummy_df], axis=1)

    # Merging all into final df, replacing all stay NaN's into 0
    cleaned_predictors.fillna(value=0, inplace=True)

    return cleaned_predictors


def main(training_data, test_data):

    # This section was there on the hypothesis that the model could be improved by first predicting NDF,
    # then destination. This might be investigated later.

    # # Creating predictor dfs for the NDF and the destination model
    # bin_preds = data_cleaner(training_data)
    # country_preds = data_cleaner(training_data[training_data['country_destination'] != 'NDF'].copy())
    #
    # # Creating the target df's for the NDF and the destination model
    # bin_target = training_data['country_destination'].copy()
    # bin_target[bin_target != 'NDF'] = 0
    # bin_target.replace(to_replace='NDF', value=1, inplace=True)
    #
    # country_target = training_data[training_data['country_destination'] != 'NDF']['country_destination'].copy()
    #
    # # Model to determine if booking was made (NDF vs actual destination)
    # model_bin = RandomForestClassifier(n_estimators=5)
    # scores = cross_validation.cross_val_score(model_bin, bin_preds, bin_target, cv=5, scoring='f1')
    # print 'Binary model scores:'
    # print scores
    #
    # # Model to determine destination if booking was made
    # model_country = RandomForestClassifier(n_estimators=5)
    # scores_country = cross_validation.cross_val_score(model_country, country_preds, country_target, cv=5, scoring='f1')
    # print 'Country model scores:'
    # print scores_country

    # Merging data to ensure consistent cleaning. Putting marker variable to separate later.
    training_data['source'] = 'training'
    test_data['source'] = 'test'
    merged_data = pd.concat([training_data, test_data])

    # Cleaning data
    cleaned_data = data_cleaner(merged_data)

    # Separating data
    pred_df = cleaned_data[cleaned_data['source'] == 'training'].copy()
    test_pred = cleaned_data[cleaned_data['source'] == 'test'].copy()

    pred_df.drop('source', axis=1, inplace=True)
    test_pred.drop('source', axis=1, inplace=True)

    target_df = training_data['country_destination'].copy()

    # Training model
    total_model = RandomForestClassifier(n_estimators=5).fit(pred_df, target_df)

    # Running the model
    predictions = total_model.predict(test_pred)

    return predictions

if __name__ == '__main__':
    train_df = pd.read_csv('train_users_2.csv', index_col='id')
    test_df = pd.read_csv('test_users.csv', index_col='id')

    output = pd.Series(main(train_df, test_df), index=test_df.index)
    output.to_csv('output.csv')
