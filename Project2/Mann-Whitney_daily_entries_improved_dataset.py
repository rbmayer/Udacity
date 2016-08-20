# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:34:39 2015
Use Mann-Whitney test to compare total subway entries by UNIT and date on rain 
and no-rain days.
"""

import numpy as np
import pandas
import scipy.stats
from pandasql import sqldf
from ggplot import *

df = pandas.read_csv('/home/rebecca/Nanodegree/Project2/improved-dataset/'
'improved-dataset/turnstile_weather_v2.csv')
        
def reduce(x):
    if x > 0:
        return 1
    else:
        return 0

# Sum entries by date and UNIT
daily_entries = (df[['DATEn','UNIT','ENTRIESn_hourly','weekday']].
                    groupby(['DATEn','UNIT','weekday']).sum())
daily_entries = daily_entries.reset_index()
# Group rain by date
daily_rain = df[['DATEn','rain']].groupby('DATEn').mean()
daily_rain = daily_rain.reset_index()
daily_rain.loc[:,'rain'] = daily_rain['rain'].apply(lambda x: reduce(x))
# Join daily_entries and daily_rain tables on date
pysqldf = lambda q: sqldf(q, globals())
q = ('''SELECT e.DATEn, e.UNIT, e.ENTRIESn_hourly, p.rain FROM daily_entries '
'e JOIN daily_rain p ON e.DATEn = p.DATEn;''')
daily_entries = pysqldf(q)
daily_entries.loc[:,'day'] = daily_entries['DATEn'].apply(lambda x: x[3:5])
# Divide daily_entries into rain and no-rain tables
no_rain = daily_entries[daily_entries.rain==0]
rain = daily_entries[daily_entries.rain==1]
# Calculate means on daily ENTRIESn_hourly
with_rain_mean = np.mean(rain['ENTRIESn_hourly'])   # result = 10863.8
without_rain_mean = np.mean(no_rain['ENTRIESn_hourly'])   # result = 10847.2

# Mann-Whitney test on entries summed by UNIT and date
# null: rain and no-rain populations have the same probability distribution
# alt: rain and no-rain populations have different distributions
# two-tailed test, p-crit = .05
U_daily, p_daily = scipy.stats.mannwhitneyu(no_rain['ENTRIESn_hourly'], 
                                            rain['ENTRIESn_hourly'])

# test results: U_daily = 5983242.5, p_daily = 0.38277629321861428
# fail to reject null; test indicates populations are not significantly different

# summarize and plot data
daily_summary = (daily_entries[['day', 'ENTRIESn_hourly', 'rain']].
                    groupby('day').sum())
daily_summary = daily_summary.reset_index()


# Mann-Whitney test on 4-hourly interval data, using precipi>0 as rain indicator

# Divide ENTRIESn_hourly into rain and no-rain tables
rain_hourly = df[df.precipi>0]
no_rain_hourly = df[df.precipi==0]

# calculate means on 4-hour ENTRIESn_hourly
with_rain_mean_hourly = np.mean(rain_hourly['ENTRIESn_hourly']) 
# result = 1743.309
without_rain_mean_hourly = np.mean(no_rain_hourly['ENTRIESn_hourly']) 
# result = 1896.74

# Mann-Whitney test
# null: rain and no-rain populations have the same probability distribution
# alt: rain and no-rain populations have different distributions
# two-tailed test, p-crit = .05
U_hourly, p_hourly = scipy.stats.mannwhitneyu(no_rain_hourly['ENTRIESn_hourly'], 
                                              rain_hourly['ENTRIESn_hourly'])

# test results U_hourly = 53838221.0, p_hourly = 9.5605890555715696e-05
# reject null; test indicates that populations are significantly different