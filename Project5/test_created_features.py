# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:33:44 2015
Test SVM algorithm using original feature set
@author: Udacity modified by rebecca
"""

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from dataset import prepare_data_transformed_and_default
from lasso import select_features_with_lasso, scale_features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Get data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# convert to dataframe for ease of handling
df = pd.DataFrame(data_dict).transpose()

# remove TOTAL row from data_dict and df
df = df.drop(['TOTAL'])
del data_dict['TOTAL']

# replace nans with 0 for all except email_address as per Sheng's comment 
# "For the financial features, the '-' symbol is assumed to be representative 
# of 0-values" (https://discussions.udacity.com/t/enron-payment-dataset-
# missing-data-or-0/19068/2)
# convert 'NaN's tp numpy nan
df = df.replace('NaN', np.nan)
# replace all NaNs in financial fields with 0
df[['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
    'exercised_stock_options', 'expenses', 'loan_advances', 
    'long_term_incentive', 'other', 'restricted_stock',  
    'restricted_stock_deferred', 'salary', 'total_payments', 
    'total_stock_value']] = df[['bonus', 'deferral_payments', 
                                'deferred_income', 'director_fees', 
                                'exercised_stock_options', 'expenses', 
                                'loan_advances', 'long_term_incentive', 
                                'other', 'restricted_stock',  
                                'restricted_stock_deferred', 'salary', 
                                'total_payments', 
                                'total_stock_value']].replace('NaN', 0)

# create dummy to indicate whether email data is available
def has_emails(x):
    if (np.isnan(x['to_messages']) & np.isnan(x['from_messages']) & 
    np.isnan(x['from_poi_to_this_person']) & 
    np.isnan(x['from_this_person_to_poi']) & 
    np.isnan(x['shared_receipt_with_poi'])):
        return 0
    else:
        return 1

df['inbox_available'] = df.apply(lambda x: has_emails(x), axis=1)

# calculate medians of email features excluding missing observations
from_msgs_median = np.median(df['from_messages']
                               [np.isnan(df.from_messages)==False])
to_msgs_median = np.median(df['to_messages'][np.isnan(df.to_messages)==False])
from_poi_median = np.median(df['from_poi_to_this_person']
                               [np.isnan(df.from_poi_to_this_person)==False])
to_poi_median = np.median(df['from_this_person_to_poi']
                               [np.isnan(df.from_this_person_to_poi)==False])
shared_poi_median = np.median(df['shared_receipt_with_poi']
                               [np.isnan(df.shared_receipt_with_poi)==False])

# impute medians in place of missing email observations
df['from_messages'] = df['from_messages'].fillna(from_msgs_median)
df['to_messages'] = df['to_messages'].fillna(to_msgs_median)
df['from_poi_to_this_person'] = (df['from_poi_to_this_person'].
                                    fillna(from_poi_median))
df['from_this_person_to_poi'] = (df['from_this_person_to_poi'].
                                    fillna(to_poi_median))
df['shared_receipt_with_poi'] = (df['shared_receipt_with_poi'].
                                    fillna(shared_poi_median))

# Get initial features and labels
labels = df['poi'].astype(float)
 # features 1: exclude email_address
features_df = df.drop(['poi', 'email_address'], axis=1)
# convert ints to float
features_df = features_df.applymap(lambda x: float(x))
feature_names = features_df.columns.values
# get scaled features
scaled_features = scale_features(features_df)
scaled_features_df = pd.DataFrame(scaled_features, 
                                  index = features_df.index, 
                                  columns=feature_names)

# Get Lasso coefficients
from sklearn.linear_model import Lasso
regression = Lasso(alpha=0.015, fit_intercept=False, max_iter=100000,
                   selection='random', random_state=52, tol=0.01)
regression.fit(scaled_features, labels)
lasso_coef = pd.DataFrame({'coefs': regression.coef_,
                           'features': feature_names})
# Select features based on coefficient value above threshold
coefficient_threshold = 0.001
selected = lasso_coef[lasso_coef['coefs'].abs() > coefficient_threshold]
# Save selected features to features_list
features_list = ['poi']
features_list.extend(selected['features'].tolist())

### Store to my_dataset for easy export below.
scaled_features_df['poi'] = df['poi']
my_data_dict = scaled_features_df.to_dict(orient='index')
my_dataset = my_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# from sklearn.pipeline import Pipeline
from sklearn import svm, grid_search
svr = svm.SVC()
# change default gamma to 1/n_features
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10, 100], 
              'gamma':[0.0625, 1, 10], 'degree':[4, 5, 6, 7, 8]}
clf_GridSearch = grid_search.GridSearchCV(svr, parameters, scoring='f1')
clf_GridSearch.fit(features_train, labels_train)
clf_GridSearch.best_estimator_

clf = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=4, gamma=0.0625, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# Results from tester.py:
'''
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=4, gamma=0.0625, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
        Accuracy: 0.88827       Precision: 0.69565      Recall: 0.28800 F1: 0.40736     F2: 0.32623
        Total predictions: 15000        True positives:  576    False positives:  252   False negatives: 1424   True negatives: 12748
'''