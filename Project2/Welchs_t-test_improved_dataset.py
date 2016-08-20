# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:00:00 2015
Run Welch's t-test for difference between rain/no-rain entries using improved
 dataset
"""
import numpy as np
import pandas
import scipy.stats

df = pandas.read_csv('/home/rebecca/Nanodegree/Project2/improved-dataset/'
'improved-dataset/turnstile_weather_v2.csv')

# Divide ENTRIESn_hourly into rain and no-rain tables
rain_hourly = df[df.precipi>0]
no_rain_hourly = df[df.precipi==0]

# calculate means on 4-hour ENTRIESn_hourly
with_rain_mean_hourly = np.mean(rain_hourly['ENTRIESn_hourly']) 
# result = 1743.309
without_rain_mean_hourly = np.mean(no_rain_hourly['ENTRIESn_hourly']) 
# result = 1896.74

#t-test of ENTRIESn_hourly
# null: mean ENTRIESn_hourly with rain equals mean ENTRIESn_hourly with no rain
# alt: mean ENTRIESn_hourly with rain does not equal mean ENTRIESn_hourly with
# no rain
# two-tailed test, p-crit = .05
t, p = scipy.stats.ttest_ind(rain_hourly['ENTRIESn_hourly'], 
                             no_rain_hourly['ENTRIESn_hourly'], 
                             equal_var=False)
# test results: t = -3.0194367408542613, p = 0.0025514015170549189
# reject null; mean ridership is significantly decreased when raining
