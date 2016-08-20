#!/usr/bin/python

import numpy as np
import pandas as pd
import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import (dump_classifier_and_data, test_classifier,
                    load_classifier_and_data)
sys.path.append("../tools/")


def use_test_classifier(clf, my_dataset, features_list):
    # test classifier using tester functions
    dump_classifier_and_data(clf, my_dataset, features_list)
    clf, dataset, feature_list = load_classifier_and_data()
    test_classifier(clf, dataset, feature_list)


### Task 1: Select what features you'll use.
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# preprocess data & eliminate missing fields through imputation
# remove TOTAL row
del data_dict['TOTAL']
# convert dict to dataframe
df = pd.DataFrame(data_dict).transpose()
# convert 'NaN' to numpy.nan
df = df.replace('NaN', np.nan)
# replace all NaNs in financial fields with 0
financial_fields = (['bonus', 'deferral_payments', 'deferred_income',
                     'director_fees', 'exercised_stock_options',
                     'expenses', 'loan_advances', 'long_term_incentive',
                     'other', 'restricted_stock',
                     'restricted_stock_deferred', 'salary',
                     'total_payments', 'total_stock_value'])
df[financial_fields] = df[financial_fields].replace('NaN', 0)
# Feature selection completed using Lasso in Task 3: Create new feature(s)

### Task 2: Remove outliers
# No outliers removed

### Task 3: Create new feature(s)
# email_available = 0 if to_messages and from_messages are NaN, else 1
df['email_available'] = df.apply(lambda x:
                                 0 if(np.isnan(x['to_messages']) &
                                      np.isnan(x['from_messages']))
                                 else 1, axis=1)
# convert email message counts into proportions
df['from_poi'] = df.apply(lambda row: np.NaN if
                          (np.isnan(row['from_messages']) &
                           np.isnan(row['to_messages'])) else
                          (row['from_this_person_to_poi'] /
                           row['from_messages']), axis=1)

df['to_poi'] = df.apply(lambda row: np.NaN if
                        (np.isnan(row['from_messages']) &
                         np.isnan(row['to_messages'])) else
                        (row['from_poi_to_this_person'] /
                         row['to_messages']), axis=1)

df['shared_poi'] = df.apply(lambda row: np.NaN if
                            (np.isnan(row['from_messages']) &
                             np.isnan(row['to_messages'])) else
                            (row['shared_receipt_with_poi'] /
                             row['to_messages']), axis=1)
# calculate medians of email features excluding missing observations
from_poi_median = np.median(df['from_poi'][np.isnan(df.from_poi) == False])
to_poi_median = np.median(df['to_poi'][np.isnan(df.to_poi) == False])
shared_poi_median = np.median(df['shared_poi'][np.isnan(df.shared_poi) ==
                              False])
# impute medians in place of missing email observations
df['from_poi'] = df['from_poi'].fillna(from_poi_median)
df['to_poi'] = df['to_poi'].fillna(to_poi_median)
df['shared_poi'] = df['shared_poi'].fillna(shared_poi_median)
# remove raw msg indicators
df.drop(['from_this_person_to_poi', 'from_poi_to_this_person',
         'to_messages', 'from_messages', 'shared_receipt_with_poi'],
        axis=1, inplace=True)

### Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Prepare features and labels for Lasso
labels = df['poi'].astype(float)
features_df = df.drop(['poi', 'email_address'], axis=1)
features_df = features_df.applymap(lambda x: float(x))
feature_names = features_df.columns.values
# Scale and center features
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(features_df)
scaled_features = scaler.transform(features_df)
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
scaled_features_df = pd.DataFrame(scaled_features, index=features_df.index,
                                  columns=features_df.columns.values)
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

    # Task 4 combined with Task 5 below

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Classifier 1: GaussianNB
# Example starting point. Try investigating other evaluation techniques!
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
use_test_classifier(clf, my_dataset, features_list)

# Classifier 2: k Nearest Neighbors with GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search
knn = KNeighborsClassifier()
parameters = {'n_neighbors': [2, 5, 10, 20, 40],
              'weights': ('distance', 'uniform'),
              'metric': ('euclidean', 'manhattan', 'chebyshev')}
knn_GridSearch = grid_search.GridSearchCV(knn, parameters, scoring='f1')
knn_GridSearch.fit(features_train, labels_train)
print ('GridSearchCV result: kNearestNeighbors best classifier is ' +
       str(knn_GridSearch.best_estimator_)) + '\n'
# Best estimator entered manually to avoid error
clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                           metric_params=None, n_jobs=1, n_neighbors=2, p=2,
                           weights='distance')
use_test_classifier(clf, my_dataset, features_list)

# Classifier 3: SVM with GridSearchCV
from sklearn import svm
svr = svm.SVC()
# change default gamma to 1/n_features
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10, 100], 'gamma':
              [0.125, 1, 10], 'degree': [4, 5, 6, 7, 8]}
svm_GridSearch = grid_search.GridSearchCV(svr, parameters, scoring='f1')
svm_GridSearch.fit(features_train, labels_train)
svm_GridSearch.best_estimator_
print ('GridSearchCV result: SVM best classifier is ' +
       str(svm_GridSearch.best_estimator_)) + '\n'
# Best estimator entered manually
clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=7, gamma=10, kernel='poly',
              max_iter=-1, probability=False, random_state=None, 
              shrinking=True, tol=0.001, verbose=False)
use_test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)