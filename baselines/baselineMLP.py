#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV    


# #### Read the dataset 
# - data normal   

# In[24]:


# import data set from the local driver 
dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
dff.head() 
df=pd.read_csv('SWaT_Dataset_Attack_v0 - Copy.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp')
frames = [dff,df] 
df_concat=pd.concat(frames) 
df_concat.head(5)     


# #### Automatic Sensor data extraction   

# In[4]:


# Extract the sensor components that begin with the list 
dff=df_concat.filter(regex='(^Time|^PIT|^AIT|^FIT|^DPI|^LIT|^Norma)',axis=1)#.head()
dff.head()  


# In[26]:


# # timestamp to DateTime 
df_concat['Timestamp'] = pd.to_datetime(df_concat['Timestamp']) 
df_concat.head()  


# In[27]:


# remove the space on Normal/Attack columns 
df_concat['Normal/Attack'] = df_concat['Normal/Attack'].str.replace(' ', '')  
#To see how the data is spread betwen Attack and Normal 
print(df_concat.groupby('Normal/Attack')['Normal/Attack'].count())  
# Rename the col name Normal/Attack with A   
df_concat.rename(columns = {'Normal/Attack':'A'}, inplace = True)
df_concat.head(2)    


# In[30]:


# Convert non-numeric class to numeric

df_concat.A[df_concat.A== 'Normal'] = 0 
df_concat.A[df_concat.A == 'Attack'] = 1    
df_concat.head()  


# In[31]:


# make the class as float 
df_concat['A'] = df_concat['A'].astype('float')   


# In[32]:


# a Timestamp as index 
dff= df_concat.set_index('Timestamp') 
dff.head()   


# In[33]:


#If there are missing entries, drop them.
dff.dropna(inplace=True)#,axis=1)  
# Total number of rows and columns 
dff.shape   


# In[34]:


# Dropping the duplicates 
dff= dff.drop_duplicates()
dff.head(2)     


# In[35]:


# looking the distribution of the data between attack and normal

print(dff.groupby('A')['A'].count())    


# In[36]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(dff['A'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(dff['A'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");    


# #### Normalize using min Max scaler    

# In[37]:


# build the scaler model 
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601'] 
scaler = MinMaxScaler() 
dff[con_feats] = scaler.fit_transform(dff[con_feats])
dff.head()   


# In[38]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

#Train test split (80/20 %) 
X_train, X_test, y_train, y_test = train_test_split(dff,dff['A'],test_size=0.2, random_state=42)
# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
  


# In[77]:


# Import MLPClassifer 
from sklearn.neural_network import MLPClassifier

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(3,4),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.1)

# Fit data onto the model
clf.fit(X_train,y_train) 


# In[73]:


# Make prediction on test dataset
ypred=clf.predict(X_test)

# Import accuracy score 
from sklearn.metrics import accuracy_score

# Calcuate accuracy
accuracy_score(y_test,ypred) 


# In[74]:


#Print memory usage 
model_memory=pd.DataFrame(ypred)

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(model_memory)  


# In[75]:


conf_matrix = confusion_matrix(y_test, ypred)  
conf_matrix


# In[76]:


print(classification_report(y_test, ypred))  


# In[49]:


# Define a function to calculate MAE and RMSE
errors = ypred - y_test
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




