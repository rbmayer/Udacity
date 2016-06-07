# -*- coding: utf-8 -*-
"""
Trial-and-error tuning of Lasso alpha parameter
 - evaluation of different values pasted in comments at bottom
Created on Wed Dec 30 13:33:44 2015
@author: Udacity modified by rebecca
"""

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


def prepare_data_with_transformed_emails(data_dict):

    # convert to dataframe for ease of handling
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(data_dict).transpose()

    # remove TOTAL row from data_dict and df
    df = df.drop(['TOTAL'])
    del data_dict['TOTAL']

    # replace nans with 0 for all except email_address as per Sheng's comment "For the financial features, the '-' symbol is assumed to be representative of 0-values" (https://discussions.udacity.com/t/enron-payment-dataset-missing-data-or-0/19068/2)
    # convert 'NaN's tp numpy nan
    df = df.replace('NaN', np.nan)
    # replace all NaNs in financial fields with 0
    df[['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock',  'restricted_stock_deferred', 'salary', 'total_payments', 'total_stock_value']] = df[['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock',  'restricted_stock_deferred', 'salary', 'total_payments', 'total_stock_value']].replace('NaN', 0)

    # create dummy to indicate whether email data is available
    def has_emails(x):
        if (np.isnan(x['to_messages']) & np.isnan(x['from_messages']) & np.isnan(x['from_poi_to_this_person']) & np.isnan(x['from_this_person_to_poi']) & np.isnan(x['shared_receipt_with_poi'])):
            return 0
        else:
            return 1

    df['inbox_available'] = df.apply(lambda x: has_emails(x), axis=1)

    # transform email features into proportions
    df['from_poi'] = df.apply(lambda row: np.NaN if (np.isnan(row['from_messages']) & np.isnan(row['to_messages'])) else row['from_this_person_to_poi']/row['from_messages'], axis=1)

    df['to_poi'] = df.apply(lambda row: np.NaN if (np.isnan(row['from_messages']) & np.isnan(row['to_messages'])) else row['from_poi_to_this_person']/row['to_messages'], axis=1)

    df['shared_poi'] = df.apply(lambda row: np.NaN if (np.isnan(row['from_messages']) & np.isnan(row['to_messages'])) else row['shared_receipt_with_poi']/row['to_messages'], axis=1)

    # calculate medians of email features excluding missing observations
    from_poi_median = np.median(df['from_poi'][np.isnan(df.from_poi)==False])
    to_poi_median = np.median(df['to_poi'][np.isnan(df.to_poi)==False])
    shared_poi_median = np.median(df['shared_poi'][np.isnan(df.shared_poi)==False])

    # impute medians in place of missing email observations
    df['from_poi'] = df['from_poi'].fillna(from_poi_median)
    df['to_poi'] = df['to_poi'].fillna(to_poi_median)
    df['shared_poi'] = df['shared_poi'].fillna(shared_poi_median)

    # remove raw msg indicators
    df.drop(['from_this_person_to_poi', 'from_poi_to_this_person', 'to_messages', 'from_messages', 'shared_receipt_with_poi'], axis=1, inplace=True)
    return df

def scale_features(features):
    # Scale and center features
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    scaler.fit(features)
    features_scaled = scaler.transform(features)
    return features_scaled

def select_features_with_lasso(features, feature_names, labels, coefficient_threshold=0.001, my_alpha=0.00001):
    import pandas as pd
    # Create Lasso regression object with parameters
    from sklearn.linear_model import Lasso
    regression = Lasso(alpha=my_alpha, fit_intercept=False, max_iter=100000, selection='random', random_state=52, tol=0.01)  # fit_intercept=False because data is already centered
    # Fit regression
    regression.fit(features, labels)
    # Evaluate coefficients
    Lasso_coef = pd.DataFrame({'coefs': regression.coef_, 'features': feature_names})
    # Select features based on coefficient value above threshold
    Best_coef = Lasso_coef[Lasso_coef['coefs'].abs() > coefficient_threshold]
    feature_list = ['poi']
    feature_list.extend(Best_coef['features'].tolist())
    return feature_list


### Prepare data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
df = prepare_data_with_transformed_emails(data_dict)
# Get initial features and labels
labels = df['poi'].astype(float)
 # features 1: exclude email_address
features_df = df.drop(['poi', 'email_address'], axis=1)
# convert ints to float
features_df = features_df.applymap(lambda x: float(x))
feature_names = features_df.columns.values
# get scaled features
scaled_features = scale_features(features_df)
scaled_features_df = pd.DataFrame(scaled_features, index = features_df.index, columns=features_df.columns.values)
# Get features
features_list = select_features_with_lasso(scaled_features, feature_names, labels, coefficient_threshold=0.001, my_alpha=0.015)

### Store to my_dataset for easy export below.
scaled_features_df['poi'] = df['poi']
my_data_dict = scaled_features_df.to_dict(orient='index')
my_dataset = my_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Evaluate feature set in SVM
from sklearn import svm, grid_search
svr = svm.SVC()
# change default gamma to 1/n_features
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10, 100], 'gamma':[0.125, 1, 10], 'degree':[4, 5, 6, 7, 8]}
clf_GridSearch = grid_search.GridSearchCV(svr, parameters, scoring='f1')
clf_GridSearch.fit(features_train, labels_train)
clf_GridSearch.best_estimator_

clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=7, gamma=10, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# Last step: run tester.py

# Results from tester.py:
''' Summary
alpha  |  F1 score
-----  |  --------
0.015  |  0.46311
0.010  |  0.40230
0.012  |  0.39464
0.040  |  0.34230
0.025  |  0.26779
'''

'''lasso alpha=0.015
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=7, gamma=10, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        Accuracy: 0.83553       Precision: 0.41002      Recall: 0.53200 F1: 0.46311     F2: 0.50212
        Total predictions: 15000        True positives: 1064    False positives: 1531   False negatives:  936   True negatives: 11469'''

'''lasso alpha=0.01
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=6, gamma=10, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        Accuracy: 0.84053       Precision: 0.40210      Recall: 0.40250 F1: 0.40230     F2: 0.40242
        Total predictions: 15000        True positives:  805    False positives: 1197   False negatives: 1195   True negatives: 1180'''

'''lasso alpha=0.025
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=6, gamma=10, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        Accuracy: 0.79073       Precision: 0.25098      Recall: 0.28700 F1: 0.26779     F2: 0.27899
        Total predictions: 15000        True positives:  574    False positives: 1713   False negatives: 1426   True negatives: 11287
'''
''' alpha=0.040
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=6, gamma=10, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        Accuracy: 0.81913       Precision: 0.33224      Recall: 0.35300 F1: 0.34230     F2: 0.34864
        Total predictions: 15000        True positives:  706    False positives: 1419   False negatives: 1294   True negatives: 11581'''
''' alpha = 0.012
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=6, gamma=10, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        Accuracy: 0.84047       Precision: 0.39939      Recall: 0.39000 F1: 0.39464     F2: 0.39184
        Total predictions: 15000        True positives:  780    False positives: 1173   False negatives: 1220   True negatives: 11827'''