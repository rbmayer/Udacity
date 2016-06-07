# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:46:37 2015

@author: rebecca
"""

import numpy as np
import pandas
import scipy.stats
import pandasql
import matplotlib.pyplot as plt

turnstile_weather = pandas.read_csv( '/home/rebecca/Nanodegree/Intro to Data Science/Lesson3/turnstile_data_master_with_weather.csv', index_col=0) 

def reduce(x):
    if x > 0:
        return 1
    else:
        return 0

def entries_histogram(df):
    
        # Sum entries by date and UNIT
    global daily_entries
    daily_entries = df[['DATEn','UNIT','ENTRIESn_hourly']].groupby(['DATEn','UNIT']).sum()
    daily_entries = daily_entries.reset_index()
    # Group rain by date
    global daily_rain
    daily_rain = df[['DATEn','rain']].groupby('DATEn').mean()
    daily_rain = daily_rain.reset_index()
    daily_rain.loc[:,'rain'] = daily_rain['rain'].apply(lambda x: reduce(x))
    # Join daily_entries and daily_rain tables on date
    from pandasql import sqldf
    pysqldf = lambda q: sqldf(q, globals())
    q = '''SELECT e.DATEn, e.UNIT, e.ENTRIESn_hourly, p.rain FROM daily_entries e JOIN daily_rain p ON e.DATEn = p.DATEn;'''
    daily_entries = pysqldf(q)
    # Divide daily_entries into rain and no-rain tables
    no_rain = daily_entries[daily_entries.rain==0]
    rain = daily_entries[daily_entries.rain==1]
    x = [no_rain['ENTRIESn_hourly'], rain['ENTRIESn_hourly']]
    # plot histogram
    plt.hist(x, bins = 23, range = (0, 60000), color=['k','m'], label=["no rain","rain"])
    plt.ylim( 0, 3000 ) 
    #plt.title("Histogram of Daily Subway Entries")
    plt.xlabel("ENTRIESn_hourly summed by date and remote unit")
    plt.ylabel("Frequency")
    legend = plt.legend()
    return plt

entries_histogram(turnstile_weather)