# !/usr/bin/env
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

__author__ = 'sebastien.genty'
__version__ = '0.1'


def data_cleaner(df):
    # Initialize dataframe
    cleaned_predictors = pd.DataFrame(index=df.index)

    # Transfering columns that need no transformation (already numeric) and source for later use
    cols_to_keep = ['timestamp_first_active', 'age', 'source']
    cleaned_predictors = pd.concat([cleaned_predictors, df[cols_to_keep]], axis=1)

    # Removing outliers in the age data (under 13 and over 95)
    cleaned_predictors.loc[cleaned_predictors['age'] > 95, 'age'] = np.nan
    cleaned_predictors.loc[cleaned_predictors['age'] < 13, 'age'] = np.nan

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
    # Merging data to ensure consistent cleaning. Putting marker variable to separate later.
    training_data['source'] = 'training'
    test_data['source'] = 'test'
    merged_data = pd.concat([training_data, test_data])

    # Cleaning data
    cleaned_data = data_cleaner(merged_data)

    # Separating data, removing marker
    pred_df = cleaned_data[cleaned_data['source'] == 'training'].copy()
    test_pred = cleaned_data[cleaned_data['source'] == 'test'].copy()

    pred_df.drop('source', axis=1, inplace=True)
    test_pred.drop('source', axis=1, inplace=True)

    # Transforming target into ints, saving the key for later transformation
    labels = LabelEncoder().fit(training_data['country_destination'])
    target_df = pd.Series(labels.transform(training_data['country_destination']), index=training_data.index)

    # Training model
    xgb_model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob',
                              subsample=0.5, colsample_bytree=0.5, seed=0)
    xgb_model.fit(pred_df.as_matrix(), target_df.tolist())

    # Running the model
    preds = xgb_model.predict_proba(test_pred.as_matrix())

    # Selecting the top 5 most likely for each respondent and stacking. 
    # This section is VERY slow and could use being optimized
    model_probs = pd.DataFrame(preds, index=test_pred.index, columns=labels.classes_)

    stacked_probs = pd.Series()
    for i in model_probs.index:
        temp = model_probs.loc[i, :]
        temp_sort = pd.DataFrame(temp.sort_values(ascending=False)[:5].index)

        temp_sort['id'] = i
        temp_sort.columns = ['country', 'id']

        stacked_probs = pd.concat([stacked_probs, temp_sort])

    # # Selecting classes with highest probabilities, compiling into list
    # ids = []
    # cts = []
    # test_ids = pd.Series(test_data.index)
    # for i in range(len(test_ids)):
    #     idx = test_data.index[i]
    #     ids += [idx] * 5
    #     cts += labels.inverse_transform(np.argsort(model_probs[i])[::-1])[:5].tolist()
    #
    # predictions = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])

    # Cleaning output and returning it
    output = stacked_probs[['id', 'country']]
    return output


if __name__ == '__main__':
    train_df = pd.read_csv('train_users_2.csv', index_col='id')
    test_df = pd.read_csv('test_users.csv', index_col='id')

    predictions = main(train_df, test_df)
    predictions.to_csv('output xgb proba.csv', index=False)
