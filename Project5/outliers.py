# -*- coding: utf-8 -*-
"""
Exploratory data analysis and outlier identification in processed Enron dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('df_processed.csv', index_col=0)

# describe processed dataset
'''
In [5]: df.describe()
Out[5]:
                bonus  deferral_payments  deferred_income  director_fees  \
count      145.000000         145.000000       145.000000     145.000000
mean    671335.303448      220557.903448   -192347.524138    9911.489655
std    1230147.632511      751704.629341    604117.425636   31202.712940
min          0.000000     -102500.000000  -3504386.000000       0.000000
25%          0.000000           0.000000    -36666.000000       0.000000
50%     300000.000000           0.000000         0.000000       0.000000
75%     800000.000000        7961.000000         0.000000       0.000000
max    8000000.000000     6426990.000000         0.000000  137864.000000

       exercised_stock_options       expenses    loan_advances  \
count               145.000000     145.000000       145.000000
mean            2061486.103448   35131.372414    578793.103448
std             4781941.261994   45247.175705   6771011.748312
min                   0.000000       0.000000         0.000000
25%                   0.000000       0.000000         0.000000
50%              607837.000000   18834.000000         0.000000
75%             1668260.000000   53122.000000         0.000000
max            34348384.000000  228763.000000  81525000.000000

       long_term_incentive            other       poi  restricted_stock  \
count           145.000000       145.000000       145        145.000000
mean         334633.986207    295210.020690  0.124138     862546.386207
std          685363.855952   1127404.270001  0.330882    2010852.212383
min               0.000000         0.000000     False   -2604490.000000
25%               0.000000         0.000000         0          0.000000
50%               0.000000       947.000000         0     360528.000000
75%          374347.000000    150458.000000         0     698920.000000
max         5145434.000000  10359729.000000      True   14761694.000000

       restricted_stock_deferred          salary  total_payments  \
count                 145.000000      145.000000    1.450000e+02
mean                72911.572414   184167.096552    2.243477e+06
std               1297469.064327   196959.768365    8.817819e+06
min              -1787380.000000        0.000000    0.000000e+00
25%                     0.000000        0.000000    9.109300e+04
50%                     0.000000   210500.000000    9.161970e+05
75%                     0.000000   269076.000000    1.934359e+06
max              15456290.000000  1111258.000000    1.035598e+08

       total_stock_value  email_available    from_poi      to_poi  shared_poi
count         145.000000       145.000000  145.000000  145.000000  145.000000
mean      2889718.124138         0.593103    0.150087    0.033032    0.630211
std       6172223.035654         0.492958    0.166962    0.031718    0.222214
min        -44093.000000         0.000000    0.000000    0.000000    0.018377
25%        221141.000000         0.000000    0.054054    0.021739    0.636146
50%        955873.000000         1.000000    0.100574    0.025845    0.661044
75%       2282768.000000         1.000000    0.198436    0.029817    0.722973
max      49110078.000000         1.000000    1.000000    0.217341    1.001145

'''

# identify outliers in each category using R boxplot definition (>1.5X IQR)
df[['bonus']][df.bonus > 1.5*(df.bonus.quantile(q=0.75)-df.bonus.quantile(q=0.25))].sort_values(by='bonus').count() # 19
df[['poi']][(df.bonus > 1.5*(df.bonus.quantile(q=0.75)-df.bonus.quantile(q=0.25))) & (df.poi==True)].count() # 9

df[['deferral_payments']][df.deferral_payments > 1.5*(df.deferral_payments.quantile(q=0.75)-df.deferral_payments.quantile(q=0.25))].sort_values(by='deferral_payments').count() # 35
df[['poi']][(df.deferral_payments > 1.5*(df.deferral_payments.quantile(q=0.75)-df.deferral_payments.quantile(q=0.25))) & (df.poi==True)].count() # 4

df[['director_fees']][df.director_fees > 1.5*(df.director_fees.quantile(q=0.75)-df.director_fees.quantile(q=0.25))].sort_values(by='director_fees').count() # 16
df[['poi']][(df.director_fees > 1.5*(df.director_fees.quantile(q=0.75)-df.director_fees.quantile(q=0.25))) & (df.poi==True)].count() # 0

df[['deferred_income']][df.deferred_income < (df.deferred_income.quantile(q=0.25)  - 1.5*(df.deferred_income.quantile(q=0.75) -df.deferred_income.quantile(q=0.25)))].sort_values(by='deferred_income').count() # 32
df[['poi']][(df.deferred_income < df.deferred_income.quantile(q=0.25)  - 1.5*(df.deferred_income.quantile(q=0.75) -df.deferred_income.quantile(q=0.25))) & (df.poi==True)].count() # 9

df[['exercised_stock_options']][df.exercised_stock_options > 1.5*(df.exercised_stock_options.quantile(q=0.75)-df.exercised_stock_options.quantile(q=0.25))].sort_values(by='exercised_stock_options').count() # 26
df[['poi']][(df.exercised_stock_options > 1.5*(df.exercised_stock_options.quantile(q=0.75)-df.exercised_stock_options.quantile(q=0.25))) & (df.poi==True)].count() # 6

df[['expenses']][df.expenses > 1.5*(df.expenses.quantile(q=0.75)-df.expenses.quantile(q=0.25))].sort_values(by='expenses').count() # 23
df[['poi']][(df.expenses > 1.5*(df.expenses.quantile(q=0.75)-df.expenses.quantile(q=0.25))) & (df.poi==True)].count() # 5

df[['loan_advances']][df.loan_advances > 1.5*(df.loan_advances.quantile(q=0.75)-df.loan_advances.quantile(q=0.25))].sort_values(by='loan_advances').count() # 3
df[['poi']][(df.loan_advances > 1.5*(df.loan_advances.quantile(q=0.75)-df.loan_advances.quantile(q=0.25))) & (df.poi==True)].count() # 1

df[['long_term_incentive']][df.long_term_incentive > 1.5*(df.long_term_incentive.quantile(q=0.75)-df.long_term_incentive.quantile(q=0.25))].sort_values(by='long_term_incentive').count() # 22
df[['poi']][(df.long_term_incentive > 1.5*(df.long_term_incentive.quantile(q=0.75)-df.long_term_incentive.quantile(q=0.25))) & (df.poi==True)].count() # 8

df[['other']][df.other > 1.5*(df.other.quantile(q=0.75)-df.other.quantile(q=0.25))].sort_values(by='other').count() # 29
df[['poi']][(df.other > 1.5*(df.other.quantile(q=0.75)-df.other.quantile(q=0.25))) & (df.poi==True)].count() # 5

df[['restricted_stock']][df.restricted_stock > 1.5*(df.restricted_stock.quantile(q=0.75)-df.restricted_stock.quantile(q=0.25))].sort_values(by='restricted_stock').count() # 25
df[['poi']][(df.restricted_stock > 1.5*(df.restricted_stock.quantile(q=0.75)-df.restricted_stock.quantile(q=0.25))) & (df.poi==True)].count() # 8

df[['restricted_stock_deferred']][df.restricted_stock_deferred > 1.5*(df.restricted_stock_deferred.quantile(q=0.75)-df.restricted_stock_deferred.quantile(q=0.25))].sort_values(by='restricted_stock_deferred').count() # 2
df[['restricted_stock_deferred']][df.restricted_stock_deferred < (df.restricted_stock_deferred.quantile(q=0.25) - 1.5*(df.restricted_stock_deferred.quantile(q=0.75)-df.restricted_stock_deferred.quantile(q=0.25)))].count() # 15
'''                  restricted_stock_deferred
BELFER ROBERT                         44093
BHATNAGAR SANJAY                   15456290'''
'''                     restricted_stock_deferred
ALLEN PHILLIP K                        -126027
BANNANTINE JAMES M                     -560222
BAY FRANKLIN R                          -82782
CARTER REBECCA C                       -307301
CHAN RONNIE                             -32460
CLINE KENNETH W                        -472568
DERRICK JR. JAMES V                   -1787380
DETMERING TIMOTHY J                    -315068
GATHMANN WILLIAM D                      -72419
HAEDICKE MARK E                        -329825
JAEDICKE ROBERT                         -44093
LOWRY CHARLES P                        -153686
NOLES JAMES L                           -94556
PIPER GREGORY F                        -409554
REYNOLDS LAWRENCE                      -14026'''
df[['poi']][(df.restricted_stock_deferred > 1.5*(df.restricted_stock_deferred.quantile(q=0.75)-df.restricted_stock_deferred.quantile(q=0.25))) & (df.poi==True)].count() # 0

df[['salary']][df.salary > 1.5*(df.salary.quantile(q=0.75)-df.salary.quantile(q=0.25))].sort_values(by='salary').count() # 11
df[['poi']][(df.salary > 1.5*(df.salary.quantile(q=0.75)-df.salary.quantile(q=0.25))) & (df.poi==True)].count() # 5

df[['total_payments']][df.total_payments > 1.5*(df.total_payments.quantile(q=0.75)-df.total_payments.quantile(q=0.25))].sort_values(by='total_payments').count() # 21
df[['poi']][(df.total_payments > 1.5*(df.total_payments.quantile(q=0.75)-df.total_payments.quantile(q=0.25))) & (df.poi==True)].count() # 4

df[['total_stock_value']][df.total_stock_value > 1.5*(df.total_stock_value.quantile(q=0.75)-df.total_stock_value.quantile(q=0.25))].sort_values(by='total_stock_value').count() # 30
df[['poi']][(df.total_stock_value > 1.5*(df.total_stock_value.quantile(q=0.75)-df.total_stock_value.quantile(q=0.25))) & (df.poi==True)].count() # 7

# 1d plots
plt.figure()
plt.plot(df['bonus'].sort(inplace=False), 'bo')
plt.ylabel('bonus')

plt.figure()
plt.plot(df['deferred_income'].sort(inplace=False), 'bo')
plt.ylabel('deferred_income')

plt.figure()
plt.plot(df['director_fees'].sort(inplace=False), 'bo')
plt.ylabel('director_fees')

plt.figure()
plt.plot(df['exercised_stock_options'].sort(inplace=False), 'bo')
plt.ylabel('exercised_stock_options')

plt.figure()
plt.plot(df['expenses'].sort(inplace=False), 'bo')
plt.ylabel('expenses')

plt.figure()
plt.plot(df['loan_advances'].sort(inplace=False), 'bo')
plt.ylabel('loan_advances')

plt.figure()
plt.plot(df['long_term_incentive'].sort(inplace=False), 'bo')
plt.ylabel('long_term_incentive')

plt.figure()
plt.plot(df['other'].sort(inplace=False), 'bo')
plt.ylabel('other')

plt.figure()
plt.plot(df['restricted_stock'].sort(inplace=False), 'bo')
plt.ylabel('restricted_stock')

plt.figure()
plt.plot(df['restricted_stock_deferred'].sort(inplace=False), 'bo')
plt.ylabel('restricted_stock_deferred')

plt.figure()
plt.plot(df[['salary', 'director_fees']].sort(columns=['salary', 'director_fees'], inplace=False))
plt.ylabel('salary/director fees') # salary has S-curve shape

plt.figure()
plt.plot(df['from_poi'].sort(inplace=False), 'bo')
plt.ylabel('from_poi')  # curve is exponential rather than normal - better to use median than mean for imputation

plt.figure()
plt.plot(df['total_payments'].sort(inplace=False), 'bo')
plt.ylabel('total_payments')  # Ken Lay is massive outlier (driven by loan_advances) but still potentially consistent with exponential distribution

plt.figure()
plt.plot(df['total_stock_value'].sort(inplace=False), 'bo')
plt.ylabel('total_stock_value')

plt.figure()
plt.plot(df['to_poi'].sort(inplace=False), 'bo')
plt.ylabel('to_poi')

plt.figure()
plt.plot(df['shared_poi'].sort(inplace=False), 'bo')
plt.ylabel('shared_poi') # looks linear, no outliers

# results: bonus, deferred_income, 'director_fees', 'exercised_stock_options', 'expenses', 'long_term_incentive', 'other', 'restricted_stock', 'salary', 'total_payments', 'total_stock_value', 'to_poi', 'from_poi': skewed data but no obvious outliers to remove. Extreme values tend to confirm exponential shape of distribution.

# outliers detected:
    # loan advances - only 3/145 > 0. Ken Lay's value is 40x higher than next closest.
    # restricted_stock_deferred - highest (BHATNAGAR SANJAY) and lowest (DERRICK JR. JAMES V) values. highest value is much farther from rest of points than lowest.
