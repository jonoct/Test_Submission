# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:59:56 2022

@author: jono
"""

#%% Required imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#%% Read and Combine Data Sets - Check Resulting Dataframes

flow_data1 = pd.read_excel('flow_1.xlsx',header=0)
flow_data2 = pd.read_excel('flow_2.xlsx',header=0)
flow_data3 = pd.read_excel('flow_3.xlsx',header=0)
flow_data4 = pd.read_excel('flow_4.xlsx',header=0)

flow_data = pd.concat([flow_data1,flow_data2,flow_data3,flow_data4],
                      ignore_index=True,
                      sort=False)

print(flow_data.head())
print()

rain_data1 = pd.read_excel('rain_1.xlsx',header=0)
rain_data2 = pd.read_excel('rain_2.xlsx',header=0)

rain_data = rain_data1.append(rain_data2, ignore_index=True)

print(rain_data.head())

#%%

## Check data types

print("\nData Types")
print("\nFlow Data")
print(flow_data.dtypes) # time data type should be an integer, not float (UNIX Time)
print("\nRain Data")
print(rain_data.dtypes) # time data type for flow data is a float, which is not equivalent to the time data type for rain data

## Describe data

print("\nData Descriptions")
print("\nFlow Data")
print(flow_data.describe()) # std is high, data values must quite spread out (check outliers)
print("\nRain Data")
print(rain_data.describe())

## Check for nulls
print("\nFlow Data Null Count")
print(flow_data.isnull().sum()) # Contains null values in both columns ('time' and 'value')
print("\nRain Data Null Count")
print(rain_data.isnull().sum()) # Contains no null values
#%% Flow Data
## Drop null values in 'time' column as they are meaningless (cannot be linked to rain data)
## Fill null values in 'value' column with average value for that column
## Rain Data set has no nulls, skip to checking for outliers
#%%
print("\nFlow Data Null Count After Dropping Null and Filling with Mean")
flow_data = flow_data.dropna(axis=0, how='all',subset=(['time']))
flow_data['value'] = flow_data['value'].fillna(flow_data['value'].mean()) ##<-- Could fill after removing outliers so that mean is not skewed
print(flow_data.isnull().sum())
#%%
# Check Shape of Data

print(flow_data.shape)

## Plot of flow data
plt.hist(flow_data['value'].values)
plt.show()


## Distribution of flow data (value)
fig = plt.figure(1, figsize=(9, 6)) 
ax = fig.add_subplot(111) 
ax.boxplot(flow_data['value'].values) 
ax.set_xticklabels(['value']) 
plt.show()

## Remove Outliers

lower_bound = 0.1 ## Set bounds as bottom and top 10%
upper_bound = 0.90
bounds = flow_data['value'].quantile([lower_bound,upper_bound])
print(bounds)
outlier_removal_index = ((flow_data['value'].values < bounds.loc[lower_bound]) | (bounds.loc[upper_bound] < flow_data['value'].values))

## Check distribution of flow data after removal
print(outlier_removal_index)
flow_data_ = flow_data.drop(flow_data.index[outlier_removal_index])
fig = plt.figure(1, figsize=(9, 6)) 
ax = fig.add_subplot(111) 
ax.boxplot(flow_data_['value'].values) 
ax.set_xticklabels(['value']) 
plt.show()

# Check Shape after outlier removal

print(flow_data_.shape)

# Check Histogram

plt.hist(flow_data_['value'].values)
plt.show()


#%%
print(rain_data.shape)

## Plot of rain data
plt.hist(rain_data['rain'].values)
plt.show()


## Distribution of rain data
fig = plt.figure(1, figsize=(9, 6)) 
ax = fig.add_subplot(111) 
ax.boxplot(rain_data['rain'].values) 
ax.set_xticklabels(['rain']) 
plt.show()

## Remove Outliers

lower_bound = 0.05 # smaller dataset, so decided to reduce bounds from 10 to 5% (std is much lower in this dataset when compared to the flow dataset)
upper_bound = 0.95
bounds = rain_data['rain'].quantile([lower_bound,upper_bound])
print(bounds)
outlier_removal_index_rain = ((rain_data['rain'].values < bounds.loc[lower_bound]) | (bounds.loc[upper_bound] < rain_data['rain'].values))


## Check distribution of rain data after removal
print(outlier_removal_index_rain)
rain_data_ = rain_data.drop(rain_data.index[outlier_removal_index_rain])
fig = plt.figure(1, figsize=(9, 6)) 
ax = fig.add_subplot(111) 
ax.boxplot(rain_data_['rain'].values) 
ax.set_xticklabels(['rain']) 
plt.show()

# Check Shape after outlier removal

print(rain_data_.shape)



plt.hist(rain_data_['rain'].values)
plt.show()
#%%
##flow_data['value'] = flow_data['value'].fillna(flow_data['value'].mean())
##print("\nFlow Data Null Count After Filling w/ Mean")
##print(flow_data.isnull().sum())


#%%
## Change flow 'time' data type from float to int <-- need time in flow data and time in rain data to be of the same type for inner join
flow_data_['time'] = flow_data_['time'].astype(np.int64)

## Check that change has been made

print("\nData Types")
print(flow_data_.dtypes) # <-- Is now the correct data type
print()
print(rain_data.dtypes)

#%%
## Group by time aggregated by average value

#%% Group tables with outliers removed

grouped_flow = flow_data_.groupby('time').value.mean()
print(grouped_flow)

grouped_rain = rain_data_.groupby('time').rain.mean()
print(grouped_rain)

#%% Group tables that still contain outliers

grouped_flow_outliers_inc = flow_data.groupby('time').value.mean()
print(grouped_flow)

grouped_rain_outliers_inc = rain_data.groupby('time').rain.mean()
print(grouped_rain)

# print(flow_data_.groupby('time').value.mean())
# print(rain_data_.groupby('time').rain.mean())

#%% Join tables (Inner join)

## Without outliers
flow_rain_data = pd.merge(grouped_flow,grouped_rain, on='time', how='inner')

print(flow_rain_data.head())

## With outliers
flow_rain_outliers_inc = pd.merge(grouped_flow_outliers_inc,grouped_rain_outliers_inc, on='time', how='inner')

print(flow_rain_outliers_inc.head())

#%% Display scatter plot
## Without Outliers

print(flow_rain_data.corr())

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(flow_rain_data['rain'], flow_rain_data['value'])
ax.set_xlabel('rain')
ax.set_ylabel('flow')
plt.show()

## Data shows an extremely weak, positive correlation and these datasets cannot be said to be correlated. 
## Fluid velocity is dictated by the physical constraints of the pipe network (diameter, elevation, head, roughness etc) and therefore
## the volume of rain might affect the rate at which it reaches a particular velocity within the pipe network, but will not be directly related to 
## fluid velocity.
## Additionally, the contribution of daily household activities to flow data adds another variable that has not been controlled for.
## The 'rain' dataset is also far too small when compared to the flow dataset, and a lot of granularity is lost when joining these two 
## datasets together.

#%% With Outliers

print(flow_rain_outliers_inc.corr())

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(flow_rain_outliers_inc['rain'], flow_rain_outliers_inc['value'])
ax.set_xlabel('rain')
ax.set_ylabel('flow')
plt.show()


# Scale data so that box plot distributions can be visualised side by side

scaler = MinMaxScaler()

flow_rain_outliers_inc = pd.DataFrame(scaler.fit_transform(flow_rain_outliers_inc),columns=['value','rain'])

# Box plots side by side

fig = plt.figure(1, figsize=(9, 6)) 
ax = fig.add_subplot(111) 
ax.boxplot(flow_rain_outliers_inc[['rain', 'value']].values) 
ax.set_xticklabels(['rain', 'flow']) 
plt.show()

## Leaving the outliers in also shows an extremely weak correlation, but this time it is negative
## It is interesing to note that what I said in the previous comment block (scatter plot without outliers), about high rainfall
## corresponding to an accellerated flow rate, can be seen in the scatter plot (high rainfall begins with a reasonbly high flowrate of 150)

