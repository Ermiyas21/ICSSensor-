#!/usr/bin/env python
# coding: utf-8

# In[136]:


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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler 
# import plotly.graph_objects as go


# from sklearn.ensemble import IsolationForest  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed 
import seaborn as sns #visualisation
  


# In[137]:


# import data set from the local driver 


df1=pd.read_csv('train1_20.csv', sep=';' )# , engine='python')
df2=pd.read_csv('train2_20.csv', sep=';' )
df3=pd.read_csv('test1_20.csv', sep=';' )
df4=pd.read_csv('test2_20.csv', sep=';' )

frames = [df1,df2,df3,df4] 
df_concat=pd.concat(frames) 
print("The dataset has {} records".format(len(df_concat)))
df_concat.head(5)

  


# In[138]:


df_concat.shape   


# In[139]:


# drop the columns that are under attack in each stage. 
df1=df_concat.drop(['attack_P1','attack_P2','attack_P3'],axis=1)#, inplace=True)     


# In[140]:


# make the Timestamp to datetime datatype 
df1['time'] = pd.to_datetime(df1['time']) # conveting the date time stamp to the timestamp series 
df1.head() 
# # df1.shape  


# In[141]:


# a Timestamp as index 
df1= df1.set_index('time') 
df1.head()   


# ### resample the data with 5 second  
# ##### Interpolation the data with each time stamp   

# In[142]:


# resample the data with 5 second 
df1=df1.resample('5s').mean()
df1.interpolate(method='polynomial',order=2)
df1.head()  


# In[143]:


# Count NaN values of whole DataFrame
nan_count = df1.isna().sum()#.sum()
print(nan_count )   


# In[144]:


df1.fillna(method = 'ffill', inplace = True)  


# In[145]:


# Visualizing The first stage with respect to time series  
def plot (): 
    plt.figure(figsize=(16, 8), dpi=150) 
    df1['P1_B3005'].plot(label='P1_B3005', color='orange') 
    df1['P1_B4005'].plot(label='P1_B4005')
    df1['P1_B3004'].plot(label='P1_B3004') 
    # adding title to the plot 
    plt.title('First Stage SWaT') 
    # adding Label to the x-axis 
    plt.xlabel('time') 
    # adding legend to the curve 
    plt.legend(title='Sensors')
    
print (plot ()) 
  


# In[146]:


#To see how the data is spread betwen Attack and Normal after interpolation 
#print(df1.groupby('Normal/Attack')['Normal/Attack'].count()) 
print(df1.groupby('attack')['attack'].count())    


# In[147]:


# make the class label into 2 since during sampling and interpolation it create the new class labels
df1.loc[df1['attack'] >= 0.1, 'lablel'] = 1 
df1.loc[df1['attack'] ==0.0, 'lablel'] = 0  


# In[148]:


# fill the missing values with forward and backward of the cols 

df1=df1.fillna(method="ffill") 


# In[149]:


print(df1.groupby('lablel')['lablel'].count())  


# In[150]:


df1= df1.drop('attack', axis=1) 


# In[151]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['lablel'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['lablel'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");    


# In[152]:


#Count 1 unique values in each columns
df2=df1.nunique().reset_index()
df2.columns = ['feature','number Of unique']
df2[df2['number Of unique']==1]   


# In[153]:



# # drop two columns name listed 
df1=df1.drop(['P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS'], axis=1)  


# In[154]:


import seaborn as sns


#get correlations of each features in dataset
c = df1.corr()
top_corr_features = c.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")   


# In[155]:


c=c.lablel.sort_values(ascending=False).head(42).keys().drop('lablel') # corelation
print(c)   


# In[156]:


# Select Multiple Columns
df2 = df1.loc[:,['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02D', 'P1_FT01', 'P1_PCV01D',
       'P1_PCV01Z', 'P1_PCV02Z', 'P1_B2016', 'P1_B2004', 'P1_FT01Z',
       'P4_ST_LD', 'P4_ST_PO', 'P4_LD', 'P3_LCV01D', 'P1_PIT01', 'P4_ST_PT01',
       'P4_HT_PO', 'P4_HT_LD', 'P1_B400B', 'P1_FT02Z', 'P1_B4005', 'P1_FT02',
       'P4_ST_FD', 'P1_PIT02', 'P1_FCV01D', 'P1_FCV01Z', 'P1_B3004',
       'P4_HT_FD', 'P1_LIT01', 'P1_B3005', 'P4_ST_PS', 'P1_FT03', 'P1_FT03Z',
       'P1_FCV02Z', 'P1_FCV02D', 'P1_LCV01Z', 'P2_VYT02', 'P1_LCV01D',
       'P3_LCP01D', 'P2_VXT02', 'P2_VXT03','lablel']]  


# #### Normalize using min Max scaler    

# In[157]:


# # For content length, use the Min max Scalar.  
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02D', 'P1_FT01', 'P1_PCV01D',
       'P1_PCV01Z', 'P1_PCV02Z', 'P1_B2016', 'P1_B2004', 'P1_FT01Z',
       'P4_ST_LD', 'P4_ST_PO', 'P4_LD', 'P3_LCV01D', 'P1_PIT01', 'P4_ST_PT01',
       'P4_HT_PO', 'P4_HT_LD', 'P1_B400B', 'P1_FT02Z', 'P1_B4005', 'P1_FT02',
       'P4_ST_FD', 'P1_PIT02', 'P1_FCV01D', 'P1_FCV01Z', 'P1_B3004',
       'P4_HT_FD', 'P1_LIT01', 'P1_B3005', 'P4_ST_PS', 'P1_FT03', 'P1_FT03Z',
       'P1_FCV02Z', 'P1_FCV02D', 'P1_LCV01Z', 'P2_VYT02', 'P1_LCV01D',
       'P3_LCP01D', 'P2_VXT02', 'P2_VXT03','lablel'] 
scaler = MinMaxScaler() 
df2[con_feats] = scaler.fit_transform(df2[con_feats])
df2.head()    


# #### Split the Data to train and Test 

# In[158]:


# Train test split (80/20 %) 
X_train, X_test, y_train, y_test = train_test_split(df2,df2['lablel'],test_size=0.2, random_state=42)
# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
  


# In[159]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)  


# #### Hyperparameter tuning for 5 second logs using Isolation forest HAI   

# In[96]:


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

# In[160]:


# Model and performance
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 


# In[161]:


# Train the isolation forest model
if_model = IsolationForest(n_estimators=5,max_features=10,max_samples=5,contamination=0.1,random_state=47).fit(X_train)
# Predict the anomalies
if_prediction = if_model.predict(X_test)
# Change the anomalies' values to make it consistent with the true values
if_prediction = [1 if i==-1 else 0 for i in if_prediction] 


# In[173]:



# 2D scatter plot for data point using IF 
from matplotlib.colors import ListedColormap

plt.xlabel("P1_FCV03D")
plt.ylabel("P1_FCV03Z") 
plt.title("Anomaly to normal data distrpution using IF ") 
x = X_test.iloc[:, 4] 
y = X_test.iloc[:, 3]
classes = ['normal', 'anomaly']
values = if_prediction 
colors = ListedColormap(['b','r'])
scatter = plt.scatter(x, y, c=values, cmap=colors)
plt.legend(handles=scatter.legend_elements()[0], labels=classes,title='data points catagory') 

plt.savefig('HAI_IF_5sed.png')  


# #### Determine anomaly score and identify anomalies  

# In[126]:


result = X_test.copy()
result['scores'] = if_model.decision_function(X_test)
result['anomaly'] = if_model.predict(X_test)
result['anomaly'] = result['anomaly'].map( {1: 0, -1: 1} )
print(result['anomaly'].value_counts())  


# In[127]:


result.head() 


# In[128]:


# filter anomaly feature which has 1 
anomaly = result.loc[result['anomaly'] == 1]
anomaly.head(5) 


# In[129]:


anomaly_index = list(anomaly.index)  


# In[130]:


raw_anomaly = df1.loc[anomaly_index]  


# In[131]:


raw_anomaly.head() 


# In[132]:


#  counts of all unique value  in FIT101 
df1["P1_B2004"].value_counts()  


# In[133]:


# count Anomalies unique values in FIT101 
raw_anomaly["P1_B2004"].value_counts() 
#raw_anomaly["FIT101"].value_counts().sort_values().plot(kind = 'barh')   


# #### perfomance evalution using RMSE   

# In[134]:


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

# In[135]:


# Check the model performance
print(classification_report(y_test, if_prediction))  


# In[ ]:




