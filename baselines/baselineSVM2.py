#!/usr/bin/env python
# coding: utf-8

# In[149]:


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


# In[150]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   


# #### Read the dataset 
# - data normal  

# In[182]:


# import data set from the local driver 
dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
#dff.head() 
df=pd.read_csv('SWaT_Dataset_Attack_v0 - Copy.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp')
frames = [dff,df] 
df_concat=pd.concat(frames) 
df_concat.head(5)    


# In[183]:


# generating another row
#df_concat = df_concat.sample(n = 100000)


# #### Automatic Sensor data extraction  

# In[184]:


# Extract the sensor components that begin with the list 
dff=df_concat.filter(regex='(^Time|^PIT|^AIT|^FIT|^DPI|^LIT|^Norma)',axis=1)#.head()
dff.head() 


# In[185]:


# remove the space on Normal/Attack columns 
dff['Normal/Attack'] = dff['Normal/Attack'].str.replace(' ', '')  
#To see how the data is spread betwen Attack and Normal 
print(dff.groupby('Normal/Attack')['Normal/Attack'].count())  
# Rename the col name Normal/Attack with A   
dff.rename(columns = {'Normal/Attack':'A'}, inplace = True)
dff.head(2)   


# In[186]:


# Convert non-numeric class to numeric

dff.A[dff.A== 'Normal'] = 0 
dff.A[dff.A == 'Attack'] = 1    
dff.head() 


# In[187]:


# make the class as float 
dff['A'] = dff['A'].astype('float')  


# In[188]:


# a Timestamp as index 
dff= dff.set_index('Timestamp') 
dff.head()  


# In[189]:


#If there are missing entries, drop them.
dff.dropna(inplace=True)#,axis=1)  
# Total number of rows and columns 
dff.shape  


# In[190]:


# Dropping the duplicates 
dff= dff.drop_duplicates()
dff.head(2)    


# In[191]:


# looking the distribution of the data between attack and normal

print(dff.groupby('A')['A'].count())   


# #### Normalize using min Max scaler   

# In[192]:


# build the scaler model 
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601'] 
scaler = MinMaxScaler() 
dff[con_feats] = scaler.fit_transform(dff[con_feats])
dff.head()   


# #### Split the Data to train and Test  

# In[194]:


#Train test split (80/20 %) 
X_train, X_test, y_train, y_test = train_test_split(dff,dff['A'],test_size=0.3, random_state=42)
# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])


    
 


# In[196]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape) 


# #### SVM Hyperparameter tuning 

# In[33]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.metrics import make_scorer, f1_score  
model = SVC() 
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear']}
f1sc = make_scorer(f1_score, average='micro')  
grid = GridSearchCV(model, param_grid, refit = True, verbose = 3)


# fitting the model for grid search
grid.fit(X_train, y_train)
# train the model on train set
best_model = grid.fit(X_train, y_train) 

print('Optimum parameters', best_model.best_params_)     


# In[197]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf',C=0.01, gamma=0.001)
svclassifier.fit(X_train, y_train) 


# In[207]:


#Predict the response for test dataset
y_pred = svclassifier.predict(X_test) 


# In[200]:


from sklearn import metrics
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred)) 


# In[206]:


#Print memory usage 
model_memory=pd.DataFrame(ypred)

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(model_memory)   


# In[205]:


# Classfication report 
print(classification_report(y_test, y_pred))   


# In[202]:


# Define a function to calculate MAE and RMSE
errors = y_pred - y_test
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors).mean()

print('The performance  of autoencoder'+ ':') 
print('')
print('Mean Absolute Error: {:.4f}'.format(mae)) 
print('Mean Square Error:{:.4f}' .format(mse))
print('Root Mean Square Error: {:.4f}'.format(rmse))
print('')   


# In[ ]:




