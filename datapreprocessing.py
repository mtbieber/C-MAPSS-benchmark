# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:25:04 2020

@author: mtbieber
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
#import plot_helper as plt_hlp
#importlib.reload(plt_hlp); # so that I can use plot_helper without reloading the notebook kernel each time 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing , HoltWintersResults
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics.pairwise import euclidean_distances

datapath = 'data/'
figpath = 'figures/'
filename = 'train_FD001.txt'

filepath = ''.join((datapath, filename))

# column names for the dataset
# op_cond refers to operational condition, sn: sensor
col_name = ['engine', 'time', 'op_cond_1', 'op_cond_2', 'op_cond_3']
col_name = col_name + ['sn_{}'.format(s + 1) for s in range(21)]


# load data into sesnor df
# notice index_col=0 is used so that the data for each separate engine can be obtained easily
#(engine columns is just a group of data)
df = pd.read_csv(filepath, header=None, names=col_name,delim_whitespace=True,index_col=0)

#Remove cols that do not change with time
col_remove=[ col for col in df.columns if (df[col].std() <= .0001*df[col].mean()) & (df[col].nunique() <=4)  ]
print('columns to be removed from analysis since they do not change with time \n',col_remove)
df.drop(columns=col_remove,axis=1,inplace=True)

#Remove columns with no apparent trend 
colsnotrend = ['op_cond_1','op_cond_2' , 'sn_9' , 'sn_14']
df.drop(colsnotrend,axis=1,inplace=True)
df.shape


# =============================================================================
# STANDARD SCALE
# =============================================================================
# get all sensors
raw_columns = df.columns.values[1:-1]
raw_sensors = df[raw_columns].values # as numpy array
print('sensors remaining for analysis after considering trends in the time series plot \n{}'.format(raw_columns))

#Standard scale
standard_scale = StandardScaler()
standard_sensors = standard_scale.fit_transform(raw_sensors)


# =============================================================================
# LINEAR REGRESSION
# =============================================================================
#fit linear regression to the sensor data to get the slopes
lin_model =LinearRegression()
engine_num=3
x = df.loc[engine_num,'RUL'].values
row_name=df.loc[engine_num].iloc[-1].name
row_sl=df.index.get_loc(row_name) # row slice to get numpy index
y=standard_sensors[row_sl] # sensor values for the specifc engine
x.reshape(-1, 1).shape
x.shape
lin_model.fit(x.reshape(-1, 1),y)
lin_model.coef_[:,0].shape

lin_model.score(x.reshape(-1, 1),y)
y_hat = lin_model.predict(x.reshape(-1, 1))
# plotting
time = df.loc[engine_num,'RUL']
cols = df.columns[1:-1]
fig, axes = plt.subplots(len(cols), 1, figsize=(19,17))
for col, ax in zip(range(standard_sensors.shape[1]), axes):
    ax.plot(time,standard_sensors[row_sl,col],label=col+1)
    ax.plot(time,y_hat[:,col],label='trend')
    ax.legend(loc=2)
fig.savefig(''.join((figpath, 'lin_trend.png')), format='png', dpi=600)

def lin_slopes(sensors,df,engine_num):
    """
    gives slopes of a teh tred lines for each sesnor 
    =================================================
    input: 
    sensors - (ndarray) numpy array of standardized signals ( rows: -RUL columns, various signals)
    engine_num - (int) engine number to selector
    df - (df) data frame of data
    output: 
    slopes -(ndarray) numpy array of slopes rows: slope of each signal linear trend line
    """
    model = LinearRegression()
    x = df.loc[engine_num,'RUL'].values
    row_name=df.loc[engine_num].iloc[-1].name
    row_sl=df.index.get_loc(row_name) # row slice to get numpy index 
    y=sensors[row_sl] # sensor values for the specifc engine
    model.fit(x.reshape(-1, 1),y)
    slopes=model.coef_[:,0]
    return slopes


# finding slopes for all engines
engines=df.index.unique().values
slopes = np.empty((standard_sensors.shape[1],len(engines)))
for i,engine in enumerate(engines):
    slopes[:,i] = lin_slopes(standard_sensors,df,engine)
    
# creating slopes_df
slopes_df = pd.DataFrame(slopes.T,index=engines,columns =raw_columns )


slopes_df.describe()
