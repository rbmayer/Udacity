# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:31:31 2015
@author: rebecca
Plot correlation map of unit R453 from the improved turnstile_weather dataset for Project 2 Sec. 3.2
"""

import numpy as np
import pandas
import pandasql
from ggplot import *

# import improved turnstile_weather dataset
df = pandas.read_csv('/home/rebecca/Nanodegree/Project2/improved-dataset/improved-dataset/turnstile_weather_v2.csv')


# create units lookup table
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
q = '''SELECT DISTINCT UNIT, station, latitude, longitude FROM df;'''
units = pysqldf(q)

# create UNIT correlation matrix
df2 = df.pivot(index='datetime', columns='UNIT', values='ENTRIESn_hourly')
unit_correlations = df2.corr()
unit_correlations = unit_correlations.reset_index()

# join UNIT correlation matrix with station names & lat/lon
unit_correlations.rename(columns={'UNIT':'unit1'}, inplace=True)    # change column label to avoid problems after join
q = '''SELECT * FROM units u join unit_correlations c ON u.UNIT = c.unit1;'''
unit_correlations = pysqldf(q)
unit_correlations = unit_correlations.drop('unit1', axis=1) # drop duplicate col

# clean up NaNs
unit_correlations = unit_correlations[unit_correlations.UNIT != 'R464']
unit_correlations = unit_correlations.drop('R464',axis=1)

# plot correlation map for R453
unit_correlations.rename(columns={'R453' : 'R453_correlations'}, inplace=True)
plotR453 = ggplot(unit_correlations[['R453_correlations','longitude','latitude']], aes(x='longitude', y='latitude', color='R453_correlations')) +geom_point() +scale_color_gradient(low='white', high='red') +ggtitle('Correlation map for 23rd St PATH station (R453)')
print plotR453