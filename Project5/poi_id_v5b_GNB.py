# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:33:44 2015
Fresh attempt to organize POI classifier
Using log-transformed financials, modified email indicators and GaussianNB
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
from dataset import prepare_data_default, prepare_data_with_transformed_emails
from lasso import select_features_with_lasso, scale_features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Get data
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
# run logistic transform on exponentially distributed features: 'bonus', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'long_term_incentive', 'other', 'restricted_stock', 'salary', 'total_payments', 'total_stock_value', 'to_poi', 'from_poi'
features_df['bonus'] = features_df['bonus'].apply(lambda x: np.log10(x+1))
features_df['deferred_income'] = features_df['deferred_income'].apply(lambda x: np.log10(x+1-min(features_df['deferred_income'])))
features_df['director_fees'] = features_df['director_fees'].apply(lambda x: np.log10(x+1))
features_df['exercised_stock_options'] = features_df['exercised_stock_options'].apply(lambda x: np.log10(x+1))
features_df['expenses'] = features_df['expenses'].apply(lambda x: np.log10(x+1))
features_df['long_term_incentive'] = features_df['long_term_incentive'].apply(lambda x: np.log10(x+1))
features_df['restricted_stock'] = features_df['restricted_stock'].apply(lambda x: np.log10(x+1-min(features_df['restricted_stock'])))
features_df['salary'] = features_df['salary'].apply(lambda x: np.log10(x+1))
features_df['total_payments'] = features_df['total_payments'].apply(lambda x: np.log10(x+1))
features_df['total_stock_value'] = features_df['total_stock_value'].apply(lambda x: np.log10(x+1-min(features_df['total_stock_value'])))
features_df['to_poi'] = features_df['bonus'].apply(lambda x: np.log10(x+1))
features_df['from_poi'] = features_df['bonus'].apply(lambda x: np.log10(x+1))
# get scaled features
scaled_features = scale_features(features_df)
scaled_features_df = pd.DataFrame(scaled_features, index = features_df.index, columns=features_df.columns.values)
# Get features
features_list = select_features_with_lasso(scaled_features, feature_names, labels, coefficient_threshold=0.001, my_alpha=0.015)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
scaled_features_df['poi'] = df['poi']
my_data_dict = scaled_features_df.to_dict(orient='index')
my_dataset = my_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# Results from tester.py:
'''GaussianNB()
        Accuracy: 0.82727       Precision: 0.25189      Recall: 0.15000 F1: 0.18803     F2: 0.16320
        Total predictions: 15000        True positives:  300    False positives:  891   False negatives: 1700   True negatives: 12109'''