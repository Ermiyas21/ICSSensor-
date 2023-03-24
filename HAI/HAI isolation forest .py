#!/usr/bin/env python
# coding: utf-8

# In[282]:


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
   


# In[283]:


# import data set from the local driver 

# df1=pd.read_csv('train1.csv')
#df1=pd.read_csv('train1_20.csv')
df1=pd.read_csv('train1_20.csv', sep=';' )# , engine='python')
df2=pd.read_csv('train2_20.csv', sep=';' )
df3=pd.read_csv('test1_20.csv', sep=';' )
df4=pd.read_csv('test2_20.csv', sep=';' )

frames = [df1,df2,df3,df4] 
df_concat=pd.concat(frames) 
print("The dataset has {} records".format(len(df_concat)))
df_concat.head(5) 


# In[284]:


df_concat.shape 


# In[285]:


# drop the columns that are under attack in each stage. 
df1=df_concat.drop(['attack_P1','attack_P2','attack_P3'],axis=1)#, inplace=True)   


# In[286]:


df1['time'] = pd.to_datetime(df1['time'])  


# In[287]:


# a Timestamp as index 
df1= df1.set_index('time') 
df1.head()  


# In[288]:


# Visualizing The first stage with respect to time series  
def plot (): 
    plt.figure(figsize=(16, 8), dpi=150) 
    df1['P1_B3005'].plot(label='P1_B3005', color='orange') 
    df1['P1_B4005'].plot(label='P1_B4005')
    df1['P1_B3004'].plot(label='P1_B3004') 
    # adding title to the plot 
    plt.title('First Stage of HAI-HIL') 
    # adding Label to the x-axis 
    plt.xlabel('Timestamp') 
    #measurment values ranges of the data point 
    plt.ylabel('ranges of the data point')
    # adding legend to the curve 
    plt.legend(title='Sensors')
    
print (plot ()) 
 


# #### Visualize the dataset 
# -  Plotting the number of normal and Attack transactions in the dataset. 

# In[289]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['attack'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['attack'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");   


# In[290]:


#Count 1 unique values in each columns
df2=df1.nunique().reset_index()
df2.columns = ['feature','number Of unique']
df2[df2['number Of unique']==1] 
 


# In[291]:


# drop the time stamp cols 
#df1= df1.drop('P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS', axis=0) 

# drop two columns name is 'C' and 'D'
df1.drop(['P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS'], axis=1) 


# In[292]:


import seaborn as sns


#get correlations of each features in dataset
c = df1.corr()
top_corr_features = c.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")   


# In[293]:


c=c.attack.sort_values(ascending=False).head(42).keys().drop('attack') # corelation
print(c) 


# In[294]:


# Select Multiple Columns
df2 = df1.loc[:, ['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02D', 'P1_FT01', 'P1_PCV02Z',
       'P1_PCV01Z', 'P1_PCV01D', 'P1_FCV02Z', 'P1_B2016', 'P1_FCV02D',
       'P1_B2004', 'P1_FT01Z', 'P1_B3005', 'P1_PIT01', 'P4_ST_PT01',
       'P2_VYT02', 'P3_LCP01D', 'P2_VXT03', 'P3_LCV01D', 'P1_B3004',
       'P2_VXT02', 'P1_FT03', 'P1_LCV01Z', 'P1_LCV01D', 'P4_HT_LD', 'P1_FT03Z',
       'P4_HT_PO', 'P4_ST_FD', 'P4_LD', 'P4_HT_FD', 'P4_ST_LD', 'P4_ST_PO',
       'P2_24Vdc', 'P2_VYT03', 'P1_TIT02', 'P1_PIT02', 'P2_VT01e', 'P1_FT02',
       'P1_FCV01Z', 'P1_FCV01D', 'P1_B400B','attack']]

 


# In[295]:


#To see how the data is spread betwen Attack and Normal 
#print(df1.groupby('Normal/Attack')['Normal/Attack'].count()) 
print(df2.groupby('attack')['attack'].count())  


# In[296]:


#df2.info() 


# In[297]:


# fill the missing values 
df2=df2.fillna(method="ffill")


# In[298]:


# # # # Remove two columns name is 'C' and 'D'
# df2=df2.drop(['time'], axis=1)  


# #### Normalize using min Max scaler  

# In[299]:


con_feats = ['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02D', 'P1_FT01', 'P1_PCV02Z',
       'P1_PCV01Z', 'P1_PCV01D', 'P1_FCV02Z', 'P1_B2016', 'P1_FCV02D',
       'P1_B2004', 'P1_FT01Z', 'P1_B3005', 'P1_PIT01', 'P4_ST_PT01',
       'P2_VYT02', 'P3_LCP01D', 'P2_VXT03', 'P3_LCV01D', 'P1_B3004',
       'P2_VXT02', 'P1_FT03', 'P1_LCV01Z', 'P1_LCV01D', 'P4_HT_LD', 'P1_FT03Z',
       'P4_HT_PO', 'P4_ST_FD', 'P4_LD', 'P4_HT_FD', 'P4_ST_LD', 'P4_ST_PO',
       'P2_24Vdc', 'P2_VYT03', 'P1_TIT02', 'P1_PIT02', 'P2_VT01e', 'P1_FT02',
       'P1_FCV01Z', 'P1_FCV01D', 'P1_B400B'] 
scaler = MinMaxScaler() 
df2[con_feats] = scaler.fit_transform(df2[con_feats])
df2.head()   


# In[300]:


#df2 = df2.reset_index()


# #### Split the Data to train and Test  

# In[301]:


# Train test split (80/20 %) 
X_train, X_test, y_train, y_test = train_test_split(df2,df2['attack'],test_size=0.2, random_state=42)
# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])


# In[302]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape) 


# #### Hyperparameter tuning for 1 second logs using Isolation forest HAI   

# In[303]:


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


# In[351]:


# Train the isolation forest model
if_model = IsolationForest(n_estimators=5,max_features=5,max_samples=10,contamination=0.1,random_state=47).fit(X_train)
# Predict the anomalies
if_prediction = if_model.predict(X_test)
# Change the anomalies' values to make it consistent with the true values
if_prediction = [1 if i==-1 else 0 for i in if_prediction] 


# In[368]:


# 2D scatter plot for data point using IF 
from matplotlib.colors import ListedColormap

plt.xlabel("P1_FCV03D")
plt.ylabel("P1_FCV03Z") 
plt.title("Anomaly to normal data distrpution using IF ") 
x = X_test.iloc[:, 0] 
y = X_test.iloc[:, 1]
classes = ['normal', 'anomaly']
values = if_prediction 
colors = ListedColormap(['b','r'])
scatter = plt.scatter(x, y, c=values, cmap=colors)
plt.legend(handles=scatter.legend_elements()[0], labels=classes,title='data points catagory') 

plt.savefig('HAI_IF_1sed.png') 


# #### Determine anomaly score and identify anomalies 

# In[307]:


result = X_test.copy()
result['scores'] = if_model.decision_function(X_test)
result['anomaly'] = if_model.predict(X_test)
result['anomaly'] = result['anomaly'].map( {1: 0, -1: 1} )
print(result['anomaly'].value_counts()) 


# In[308]:


result.head()


# In[309]:


# filter anomaly feature which has 1 
anomaly = result.loc[result['anomaly'] == 1]
anomaly.head(5)


# In[310]:


anomaly_index = list(anomaly.index) 


# In[311]:


raw_anomaly = df1.loc[anomaly_index] 


# In[313]:


raw_anomaly.head()


# In[314]:


#  counts of all unique value  in FIT101 
df1["P1_B2004"].value_counts() 


# In[315]:


# count Anomalies unique values in FIT101 
raw_anomaly["P1_B2004"].value_counts() 
#raw_anomaly["FIT101"].value_counts().sort_values().plot(kind = 'barh')  


# #### perfomance evalution using RMSE  

# In[316]:


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

# In[317]:


# Check the model performance
print(classification_report(y_test, if_prediction))  


# In[ ]:




