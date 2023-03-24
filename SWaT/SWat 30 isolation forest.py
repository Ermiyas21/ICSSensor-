#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt     

from tensorflow import keras
from sklearn.preprocessing import StandardScaler 
#import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed 
import seaborn as sns #visualisation
   


# In[80]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   


# In[81]:


# import data set from the local driver 
dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
df=pd.read_csv('SWaT_Dataset_Attack_v0 - Copy.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp')
frames = [dff,df] 
df_concat=pd.concat(frames) 
df_concat.head(5)    


# #### Automatic Sensor data extraction 

# In[82]:


# df_concat.filter(regex=('^PIT' and '^LIT') ,axis=1).head() 
df1=df_concat.filter(regex='(^Time|^PIT|^AIT|^FIT|^DPI|^LIT|^Norma)',axis=1)#.head()
df1.head() 


# In[83]:


# Convert timestamp to datetime format 
df1['Timestamp'] = pd.to_datetime(df1['Timestamp']) 

# # df1.shape

#df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
#df1 = df1.set_index('Timestamp').resample('H').mean().reset_index()
# df1.shape
df1.head()  


# In[84]:


# remove the space on Normal/Attack columns 
df1['Normal/Attack'] = df1['Normal/Attack'].str.replace(' ', '')  
#To see how the data is spread betwen Attack and Normal 
print(df1.groupby('Normal/Attack')['Normal/Attack'].count())  
# Rename the col name Normal/Attack with A   
df1.rename(columns = {'Normal/Attack':'A'}, inplace = True)
df1.head(2)  
   


# In[85]:


# Convert non-numeric class to numeric

df1.A[df1.A== 'Normal'] = 0 
df1.A[df1.A == 'Attack'] = 1    
df1.head()  


# In[86]:


# make the class as float 
df1['A'] = df1['A'].astype('float')   


# In[87]:


# aTimestamp as index 
df1= df1.set_index('Timestamp') 
df1.head()


# ### resample the data with 30 second 
# ##### Interpolation the data with each time stamp 

# In[88]:


df1=df1.resample('30s').mean()
df1.interpolate(method='polynomial',order=2)
df1.head()   


# In[89]:


df1.shape  


# In[90]:


# Count NaN values of whole DataFrame
nan_count = df1.isna().sum()#.sum()
print(nan_count )  


# In[91]:


df1.fillna(method = 'ffill', inplace = True) 


# In[92]:


# code
# Visualizing The first stage with respect to time series 
  
# to set the plot size
plt.figure(figsize=(16, 8), dpi=150)
  
# using plot method to plot open prices.
# in plot method we set the label and color of the curve.

df1['LIT101'].plot(label='LIT101', color='orange')
df1['AIT201'].plot(label='AIT201')

df1['LIT301'].plot(label='LIT301')

#df1.plot(subplots=True) 
  
# adding title to the plot
plt.title('First Stage SWaT')
  
# adding Label to the x-axis
plt.xlabel('Timestamp')
  
# adding legend to the curve
#plt.legend()
plt.legend(title='Sensors')
  


# In[93]:


#To see how the data is spread betwen Attack and Normal after interpolation 
#print(df1.groupby('Normal/Attack')['Normal/Attack'].count()) 
print(df1.groupby('A')['A'].count())   


# In[94]:


# make the class label into 2 since during sampling and interpolation it create the new class labels
df1.loc[df1['A'] >= 0.1, 'lablel'] = 1 
df1.loc[df1['A'] ==0.0, 'lablel'] = 0  


# In[95]:


# fill the missing values with forward and backward of the cols 

df1=df1.fillna(method="ffill") 


# In[96]:


print(df1.groupby('lablel')['lablel'].count())  


# In[97]:


df1= df1.drop('A', axis=1) 


# In[98]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['lablel'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['lablel'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");   


# #### Normalize using min Max scaler   

# In[99]:


# # For content length, use the Min max Scalar.  
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601'] 
scaler = MinMaxScaler() 
df1[con_feats] = scaler.fit_transform(df1[con_feats])
df1.head()   


# #### Split the Data to train and Test 

# In[100]:


# Train test split (80/20 %) 
X_train, X_test, y_train, y_test = train_test_split(df1,df1['lablel'],test_size=0.2, random_state=42)
# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
 


# In[101]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape) 


# #### Hyperparameter tuning for 30 second logs using Isolation forest for SWaT 

# In[102]:


from sklearn import model_selection 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer, f1_score 
clf = IsolationForest(random_state=47)

param_grid = {'n_estimators': [5,10,20,30,40],#list(range(100, 800)),#, 5)), 
              'max_samples': [5,10,20,30,40], #list(range(100, 500)),#, 5)), 
              'contamination': [0.1, 0.2, 0.3, 0.4],# 0.5], 
              'max_features': [5,10,15], 
              #'bootstrap': [True, False], 
              #'n_jobs': [5, 10, 20, 30]
             }

f1sc = make_scorer(f1_score, average='micro')

grid_search = model_selection.GridSearchCV(clf, 
                                                 param_grid,                                                  
                                                 refit=True,
                                                 scoring=f1sc,
                                                 cv=10, 
                                                 return_train_score=True)
#grid_dt_estimator.fit(X_train, X_test)

best_model = grid_search.fit(X_train.values, y_train) 

print('Optimum parameters', best_model.best_params_)    


# #### Train Isolation Forest Model  

# In[31]:


# Model and performance
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 


# In[103]:


# Train the isolation forest model
if_model = IsolationForest(n_estimators=5,max_samples=10,contamination=0.1,random_state=47).fit(X_train)
# Predict the anomalies
if_prediction = if_model.predict(X_test)
# Change the anomalies' values to make it consistent with the true values
if_prediction = [1 if i==-1 else 0 for i in if_prediction]
 


# In[104]:


# visualize the anomaly points in the dataset with 2D
plt.figure(figsize=(10,6)) 
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=if_prediction)
plt.title("Anomaly to normal data distrpution using IF ")
plt.xlabel("FIT101")
plt.ylabel("LIT101")
plt.show() 


# #### Determine anomaly score and identify anomalies  

# In[105]:


result = X_test.copy()
result['scores'] = if_model.decision_function(X_test)
result['anomaly'] = if_model.predict(X_test)
result['anomaly'] = result['anomaly'].map( {1: 0, -1: 1} )
print(result['anomaly'].value_counts())  


# In[106]:


# filter anomaly feature which has 1 
anomaly = result.loc[result['anomaly'] == 1]
anomaly.head(5) 


# In[107]:


anomaly_index = list(anomaly.index)  


# In[108]:


raw_anomaly = df1.loc[anomaly_index]  


# In[109]:


raw_anomaly.head()
 


# In[110]:


#  counts of all unique value  in FIT101 
df1["FIT101"].value_counts()


# In[111]:


# count Anomalies unique values in FIT101 
raw_anomaly["FIT101"].value_counts() 
#raw_anomaly["FIT101"].value_counts().sort_values().plot(kind = 'barh')  


# #### perfomance evalution using RMSE  

# In[112]:


# Define a function to calculate MAE and RMSE
errors = if_prediction - y_test
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors).mean()

print('The performance  of isolation forest'+ ':') 
print('')
print('Mean Absolute Error: {:.4f}'.format(mae)) 
print('Mean Square Error:{:.4f}' .format(mse))
print('Root Mean Square Error: {:.4f}'.format(rmse))
print('')  


# #### Confusion matrix   

# In[113]:


# Check the model performance
print(classification_report(y_test, if_prediction))  


# In[ ]:




