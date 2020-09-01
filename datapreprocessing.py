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
# ADD RUL
# =============================================================================
# add RUL to each engine based on time column, 
# notice that RUL is negative quantity here to make 0 as the end of life for all engines
for id in df.index.unique():
    df.loc[id,'RUL'] = df.loc[id]['time'].apply(lambda x: x-df.loc[id]['time'].max())
    
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


# index of highest to lowest abs(slope) for each signal 
slope_order_idx=np.argsort(np.abs(slopes.mean(axis=1)))[::-1]

# =============================================================================
# #PCA
# =============================================================================
pca = PCA()
pca.fit(standard_sensors)
100*pca.explained_variance_ratio_

#PCA with highest 6 sensors in terms of linear trend slope
num_high_slopes = 6
pca_high_n_components=3
sensors_high_trend=standard_sensors[:,slope_order_idx[0:num_high_slopes]]
pca_high = PCA(pca_high_n_components,whiten=True)
pca_high.fit(sensors_high_trend)
pca_high.explained_variance_ratio_

sensors_pca=pca_high.transform(sensors_high_trend)
sensors_pca.shape


'''
based on the analysis of linear trend, the top 6 sensors are chosen based on the magnitude of their linear trend, i.e. the magnitude of their linear regression slope. it looks that based on these 6 sensors, taking 3 first principle components s captures about 90% of the data variability. hence the further reduction in dimensionality comes at a low loss of information.

the conclusion from the above ( data exploration and processing) steps are the following:

the sensors that do not change with time ( do not have variation with engine operational cycles) are dropped since they do not offer any information toward prediction the end of life
the sensors that do not have apparent trend (looks like noise only, or do not have a trend toward the end of life) are dropped as well. this contains the sensors that behave differently for different engines ( since these will confuse the learning algorithm and can cause large testing errors since their behavior are not universal concerning all engines)
based on linear regression of the remain sensor data with RUL, the highest 6 sensors in terms of the absolute values of the slopes are kept only. these sensors change predictably at the end of life for the engines.
further, reduce the dimensionality by taking the first 3 principal components for the data
the remaining 3 components of the data will be fused to make a Health Index (HI) function with RUL for each engine
'''

# =============================================================================
# Data Preparation
# =============================================================================

# create a dictionary with engine slices 

engines=df.index.unique().values # engine numbers
engine_slices = dict()# key is engine number, value is a slice that gives numpy index for the data that pertains to an engine  

for i,engine_num in enumerate(engines):
    row_name=df.loc[engine_num].iloc[-1].name
    row_sl=df.index.get_loc(row_name) # row slice to get numpy index 
    engine_slices[engine_num]=row_sl


# create RUL vector
RUL = np.empty(len(engines))

for i,engine_num in enumerate(engines):
    RUL[i]=-1*df.loc[engine_num]['RUL'].min()
    
# ax = plt.subplot(figsize=(15,12))
fig=plt.figure(figsize=(15,12))
ax=sns.distplot(RUL)
ax.set_title('Distribution of RUL for all engines',{'fontsize':16})
ax.set_xlabel('RUL')


fig=plt.figure(figsize=(6,5))
ax=sns.distplot(RUL)
ax.set_title('Box plot of RUL for all engines',{'fontsize':16});
ax.set_ylabel('RUL');
ax = sns.boxplot( data=RUL)
ax = sns.swarmplot( data=RUL, color=".25")

(RUL>350).sum()


# =============================================================================
# HI
# =============================================================================

# conditions and thersholds for high HI and low HI
RUL_high = 300 # threshold of number of cycles that makes us consider the engine started at perfect health 
RUL_low = 5  # threshold of the number of cycles below which engine is considered has failed l ( for purposes of modeling and getting data)  
RUL_df = df['RUL'].values


# Gather data and prepare it for HI fusing and modeling


# find engines with high (low) HI at their initial (final) cycles
idx_high_HI = [RUL_df<=-RUL_high][0]
idx_low_HI  = [RUL_df>-RUL_low][0]

# data for to make fuse sensor model (HI creation)
high_HI_data= sensors_pca[idx_high_HI,:]
low_HI_data= sensors_pca[idx_low_HI,:]
# concatenate high HI and Low HI data
X_HI = np.concatenate((high_HI_data,low_HI_data),axis=0)

# target for the fused signal [ just 0 or 1 for failed ans healthy]
y_one = np.ones(high_HI_data.shape[0])
y_zero = np.zeros(low_HI_data.shape[0])
# concatenate high HI and Low HI target
y_HI = np.concatenate((y_one,y_zero),axis=0)



# linear regression
HI_linear = LinearRegression()
HI_linear.fit(X_HI,y_HI)

# logistic regression
HI_logistic = LogisticRegression(solver='liblinear')
HI_logistic.fit(X_HI,y_HI)


# get data for and engine
engine_num=50
engine_sensors=sensors_pca[engine_slices[engine_num],:]
RUL_engine = df.loc[engine_num]['RUL']

# predict the HI
HI_pred_lin = HI_linear.predict(engine_sensors)
HI_pred_log = HI_logistic.predict_proba(engine_sensors)[:,1]

# plot fused HI signal for linear and logistic models \
fig=plt.figure(figsize=(15,7))
plt.plot(RUL_engine,HI_pred_lin,label='Linear')
plt.plot(RUL_engine,HI_pred_log,label='Logistic')
plt.title('Health Index (HI)')
plt.xlabel('RUL [cycles]')
plt.ylabel('HI [-]')
plt.legend()

fig.savefig(''.join((figpath, 'HI_Log_Lin.png')), format='png', dpi=600)


# plot HI for all engines 
fig=plt.figure(figsize=(15,12))
for engine_num in engines:
    
    engine_sensors=sensors_pca[engine_slices[engine_num],:]
    RUL_engine = df.loc[engine_num]['RUL']

    # predict the HI
    HI_pred_lin = HI_linear.predict(engine_sensors)
    plt.scatter(RUL_engine,HI_pred_lin,label=engine_num,s=3)
    
    #HI_pred_log = HI_logistic.predict_proba(engine_sensors)[:,1]
    #plt.scatter(RUL_engine,HI_pred_log,label='Logistic')
    
    plt.title('Health Index (HI)')
    plt.xlabel('RUL [cycles]')
    plt.ylabel('HI [-]')
    #plt.legend()
    
