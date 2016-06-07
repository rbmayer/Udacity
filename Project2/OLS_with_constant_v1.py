# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:41:51 2015

@author: rebecca
haversine function author: @ballsdotballs, http://tinyurl.com/q4egllb

Script for linear regression of improved dataset with constant
wc_results15: remove 'hour' from features 
really final Proj2 model this time
"""

import numpy as np
import pandas
import scipy.stats
import matplotlib.pyplot as plt
from ggplot import *
import statsmodels.api as sm
import pandasql
import csv

df = pandas.read_csv('/home/rebecca/Nanodegree/Project2/improved-dataset/improved-dataset/turnstile_weather_v2.csv')

# path to save results file
path = '/home/rebecca/Nanodegree/Project2/OLSresults/wc_results15.txt'

# create holiday column in improved dataset where weekends & holidays==1, weekdays==0
def create_holiday_col(row):
    if row['DATEn'][3:5] == '30':
        return 1
    elif row['day_week'] > 4:
        return 1
    else:
        return 0
        
df.loc[:,'holiday'] = df.apply(create_holiday_col, axis=1)

# create modified day_week column in improved dataset so that Memorial Day is encoded as a Sunday
def create_day_week_mod_col(row):
    if row['DATEn'][3:5] == '30':
        return 6
    else:
        return row['day_week']
        
df.loc[:,'day_week_mod'] = df.apply(create_day_week_mod_col, axis=1)

# create 4-hour entries lag term. Fill missing entries with current entries.
df['lag_4hr'] = df['ENTRIESn_hourly'].shift(1)
df['lag_4hr'] = df['lag_4hr'].fillna(df['ENTRIESn_hourly'])

# create 8-hour entries lag term. Fill missing entries with current entries.
df['lag_8hr'] = df['ENTRIESn_hourly'].shift(2)
df['lag_8hr'] = df['lag_8hr'].fillna(df['ENTRIESn_hourly'])

# Create neighbor_count column in improved dataset

# create lookup table units
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
q = '''SELECT DISTINCT UNIT, station, latitude, longitude FROM df;'''
units = pysqldf(q)

# implement haversine function
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees). All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c # mean radius of earth = 6,371km
    # mi = km * 0.621371192
    return km
    
# create uxu matrix of distances between units
q2 = '''SELECT a.UNIT AS unit1, a.station AS station1, a.latitude AS latitude1, a.longitude AS longitude1, b.UNIT AS unit2, b.station AS station2, b.latitude AS latitude2, b.longitude AS longitude2 FROM units a JOIN units b WHERE unit1 != unit2;'''
uxu = pysqldf(q2)
uxu['dist'] = haversine(uxu['longitude1'],uxu['latitude1'],uxu['longitude2'],uxu['latitude2'])

# add neighbor count to units
def neighbor_count(unit, uxu, radius):
    # input unit ID, distance table & neighbor radius
    # output count of unit neighbors within radius
    neighbor_station_slice = uxu[(uxu.unit1==unit) & (uxu.dist<=radius)]
    count = neighbor_station_slice['unit2'].count()
    return count

radius = 1.0 # neighbor radius in km
units.loc[:,'neighbor_count'] = units['UNIT'].apply(lambda x: neighbor_count(x, uxu, radius)) 

# add neighbor count column to df (slow)
def lookup_neighbor_count(x):
    count = int(units['neighbor_count'][units.UNIT==x].values)
    return count
    
df.loc[:,'neighbor_count'] = df['UNIT'].apply(lambda x: lookup_neighbor_count(x))

# create neighbor_count x precipi interaction term
def neighbor_interaction(row):
    return row['neighbor_count'] * row['precipi']

df.loc[:,'neighbor_precipi'] = df.apply(neighbor_interaction, axis=1)

# create neighbor_count x hour interaction term
def neighbor_hour_interaction(row):
    return row['neighbor_count'] * row['hour']
    
df.loc[:,'neighbor_hour'] = df.apply(neighbor_hour_interaction, axis=1)

# take log of ENTRIESn_hourly
def take_log(x):
    if x==0:
        return 0
    else:
        return np.log(x)
        
df.loc[:,'entries_log'] = df['ENTRIESn_hourly'].apply(lambda x: take_log(x))

# Optional: save copy of df to csv
# df.to_csv('/home/rebecca/Nanodegree/Project2/improved df versions.csv', index=False, index_label=False)

# statsmodel OLS no constant
def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    rsquared = results.rsquared
    summary = results.summary()
    return intercept, params, summary
    
# define features and dummy variables
features = df[['holiday', 'precipi', 'lag_4hr', 'lag_8hr', 'neighbor_hour']]
dummy_units = pandas.get_dummies(df['UNIT'], prefix='unit')
features = features.join(dummy_units)
# define values
values = df['ENTRIESn_hourly']
# run OLS
intercept, params, summary = linear_regression(features, values)
# print summary to text file
f = open(path, 'w')
summary = str(summary)
f.write(summary)
f.close()

# calculate and plot residuals
def plot_residuals(features, intercept, params, values):
    predictions = intercept + np.dot(features, params)
    residuals = values - predictions
    plt.figure()
    residuals.hist(bins=50)
    return plt
    
print plot_residuals(features, intercept, params, values)
