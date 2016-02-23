#! bin/var/env
import numpy as np
import pandas as pd
import xgboost as xgb


__author__ = 'sebastien.genty'
__version__ = '1.0'


def data_cleaner(df):
    # Separating numerical and other data...
    print 'Separating by type...'
    cleaned_df = df.select_dtypes(exclude=[object])
    cat_data = df.select_dtypes(include=[object])
    cat_data.drop(['marker'], inplace=True, axis=1)

    # Dealing with pesky V22
    print 'Recoding v22...'
    v22_counts = df['v22'].value_counts()[:15]
    cat_data.loc[~df['v22'].isin(v22_counts.index), 'v22'] = np.nan

    # Dummying the object columns
    print 'Dummyfying and joining the other categories...'
    for i in cat_data:
        print '     Column ' + str(i)
        temp_dummy_df = pd.get_dummies(cat_data[i], prefix=i, sparse=True)
        cleaned_df = pd.concat([cleaned_df, temp_dummy_df], axis=1)

    cleaned_df = pd.concat([cleaned_df, df['marker']], axis=1)

    return cleaned_df


def data_separator(df, value):
    # Selecting right subset of data, removing target and marker variable
    output = df[df['marker'] == value].copy()
    output.drop(['target', 'marker'], axis=1, inplace=True)

    return output


def xgboost_predictor(train, test):
    # Marking data and combining for even cleaning
    print 'Merging the data...'
    train['marker'] = 'train'
    test['marker'] = 'test'
    combined = pd.concat([train, test])

    print 'Cleaning the data...'
    cleaned_combined_df = data_cleaner(combined)

    # Separating the data, dropping the marker and target in feature df
    print 'Formatting the data for model training and predictions...'
    target = train['target']
    test_features = data_separator(cleaned_combined_df, 'test')
    train_features = data_separator(cleaned_combined_df, 'train')

    # Training model
    print 'Model training begins...'
    xgb_model = xgb.XGBClassifier(max_depth=7, learning_rate=0.3, n_estimators=50, objective='binary:logistic',
                                  subsample=0.5, colsample_bytree=0.5, seed=0, silent=0)
    xgb_model.fit(train_features.as_matrix(), target.tolist())

    # Running the model
    print 'Making predictions....'
    preds = xgb_model.predict_proba(test_features.as_matrix())

    print 'Cleaning predictions to match expected format....'
    output = pd.DataFrame(preds, index=test.index)
    output.drop(0, axis=1, inplace=True)
    output.columns = ['PredictedProb']

    return output


def main():
    # Loading the data
    print 'Loading the data...'
    train_df = pd.read_csv('train.csv', index_col=0)
    test_df = pd.read_csv('test.csv', index_col=0)
    preds = xgboost_predictor(train_df, test_df)

    print 'Saving predictions...'
    preds.to_csv('predictions.csv')

if __name__ == '__main__':
    main()
