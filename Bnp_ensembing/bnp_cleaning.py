#! bin/var/env python2.7
import pandas as pd

from datetime import datetime

__author__ = 'sebastien.genty'
__version__ = '1.0'


def data_cleaner(df, dummy=False):
    pd.options.mode.chained_assignment = None
    # Separating numerical and other data...
    print 'Separating by type...'
    cleaned_df = df.select_dtypes(exclude=[object])
    cat_data = df.select_dtypes(include=[object])
    cat_data.drop(['marker'], inplace=True, axis=1)

    if dummy:
        # Dealing with pesky V22
        print 'Recoding v22...'
        cat_data.drop(['v22'], inplace=True, axis=1)
        # v22_counts = df['v22'].value_counts()[:15]
        # cat_data.loc[~df['v22'].isin(v22_counts.index), 'v22'] = np.nan

        # Dummying the object columns
        print 'Dummyfying and joining the other categories...'
        cat_data_something = pd.DataFrame(index=cat_data.index)
        for i in cat_data:
            print '     Column ' + str(i)
            temp_dummy_df = pd.get_dummies(cat_data[i], prefix=i)
            cat_data_something = pd.concat([cat_data_something, temp_dummy_df], axis=1)

        cleaned_df = pd.concat([cleaned_df, cat_data_something], axis=1)

    else:
        # Factoring the object columns
        print 'Factoring and joining the other categories...'
        for i in cat_data:
            print '     Column ' + str(i)
            temp_dummy_df, _ = pd.factorize(cat_data[i])
            temp_dummy_df = pd.DataFrame(temp_dummy_df, index=cat_data.index, columns=[i])
            cleaned_df = pd.concat([cleaned_df, temp_dummy_df], axis=1)

    cleaned_df = pd.concat([cleaned_df, df['marker']], axis=1)

    return cleaned_df


def data_separator(df, value):
    # Selecting right subset of data, removing target and marker variable
    output = df[df['marker'] == value].copy()
    output.drop(['marker'], axis=1, inplace=True)

    output.fillna(value=0, inplace=True)
    return output


def data_formater(train, test, dummy=False):
    # Marking data and combining for even cleaning
    print 'Merging the data...'
    train['marker'] = 'train'
    test['marker'] = 'test'
    combined = pd.concat([train, test])

    print 'Cleaning the data...'
    cleaned_combined_df = data_cleaner(combined, dummy)

    # Separating the data, dropping the marker and target in feature dfs except for the training one
    print 'Formatting the data for model training and predictions...'
    test_features = data_separator(cleaned_combined_df, 'test')
    train_features = data_separator(cleaned_combined_df, 'train')
    test_features.drop('target', axis=1, inplace=True)

    print 'Turning the target into a categorical...'
    train_cat = train_features.copy()
    train_cat['target'].replace(to_replace=1, value='YES', inplace=True)
    train_cat['target'].replace(to_replace=0, value='NO', inplace=True)

    return train_features, train_cat, test_features


def main():
    # Printing start time to know how long this thing's been going on for
    print datetime.now().time().isoformat()

    # Loading the data
    print 'Loading the data...'
    train = pd.read_csv('../BNP/train.csv', index_col=0)
    test = pd.read_csv('../BNP/test.csv', index_col=0)

    # Actual work
    train_uncat_cleaned, train_cat_cleaned, test_cleaned = data_formater(train, test)

    # Saving the data for later use
    print 'Saving the data...'
    train_uncat_cleaned.to_csv('training_uncat_df.csv', header=True)
    train_cat_cleaned.to_csv('training_cat_df.csv')
    test_cleaned.to_csv('test_df.csv')
    train_uncat_cleaned.drop('target', axis=1).to_csv('training untagged.csv')

    print 'Getting the list of predictors for models and other things...'
    pred_list = pd.Series(test.columns)
    pred_list.to_csv('pred list.csv')

    print 'All is well that ends well. Thank you for using this program!'

if __name__ == '__main__':
    main()
