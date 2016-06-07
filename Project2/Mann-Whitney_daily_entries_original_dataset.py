# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:34:39 2015

@author: rebecca

Use Mann-Whitney test to compare total subway entries by UNIT and date on rain and no-rain days.
"""

import numpy as np
import pandas
import scipy.stats
import pandasql
from ggplot import *

turnstile_weather = pandas.read_csv( '/home/rebecca/Nanodegree/Intro to Data Science/Lesson3/turnstile_data_master_with_weather.csv', index_col=0) 

# Sum entries by date and UNIT
daily_entries = turnstile_weather[['DATEn','UNIT','ENTRIESn_hourly']].groupby(['DATEn','UNIT']).sum()
daily_entries = daily_entries.reset_index()
# Summarize rain by date
daily_rain = turnstile_weather[['DATEn','rain']].groupby('DATEn').mean()
daily_rain = daily_rain.reset_index()
# Join daily_entries and daily_rain tables on date
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
q = '''SELECT e.DATEn, e.UNIT, e.ENTRIESn_hourly, p.rain FROM daily_entries e JOIN daily_rain p ON e.DATEn = p.DATEn;'''
daily_entries = pysqldf(q)
daily_entries.loc[:,'day'] = daily_entries['DATEn'].apply(lambda x: x[8:10])
# Divide daily_entries into rain and no-rain tables
no_rain = daily_entries[daily_entries.rain==0]
rain = daily_entries[daily_entries.rain==1]
# Calculate means
with_rain_mean = np.mean(rain['ENTRIESn_hourly'])  # result = 10502.9
without_rain_mean = np.mean(no_rain['ENTRIESn_hourly']) # result = 10332.0

# Perform Mann-Whitney test
U, p = scipy.stats.mannwhitneyu(no_rain['ENTRIESn_hourly'],rain['ENTRIESn_hourly'])

# test results: U = 21345225.5, p = 0.2227
