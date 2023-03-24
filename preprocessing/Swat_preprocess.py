#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[307]:


# ! pip install pyanom
# # ! conda install tensorflow -y


# In[3]:


# ! conda install  pyanom


# #### Import required libraries 

# In[4]:


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


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 


# #### Read the dataset 
# - data integration both normal and attack 

# In[6]:


# import data set from the local driver 
dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
df=pd.read_csv('SWaT_Dataset_Attack_v0 - Copy.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp')
frames = [dff,df] 
df_concat=pd.concat(frames) 
df_concat.head(5)  


# In[7]:


# df_concat.to_csv(r'/home/jovyan/SWaT dataset.csv',index=False) 


# #### Automatic Sensor data extraction 

# In[8]:


# Extract the sensor components that begin with the list 
df1=df_concat.filter(regex='(^Time|^PIT|^AIT|^FIT|^DPI|^LIT|^Norma)',axis=1)#.head()
df1.head()


# In[9]:


# # timestamp to DateTime 
df1['Timestamp'] = pd.to_datetime(df1['Timestamp']) 
df1.head()


# In[10]:


# remove the space on Normal/Attack columns 
df1['Normal/Attack'] = df1['Normal/Attack'].str.replace(' ', '')  
#To see how the data is spread betwen Attack and Normal 
print(df1.groupby('Normal/Attack')['Normal/Attack'].count())  
# Rename the col name Normal/Attack with A   
df1.rename(columns = {'Normal/Attack':'A'}, inplace = True)
df1.head(2)  


# In[11]:


# Convert non-numeric class to numeric

df1.A[df1.A== 'Normal'] = 0 
df1.A[df1.A == 'Attack'] = 1    
df1.head()


# In[12]:


# make the class as float 
df1['A'] = df1['A'].astype('float') 


# In[13]:


# a Timestamp as index 
df1= df1.set_index('Timestamp') 
df1.head()


# In[14]:


df1.shape


# In[214]:


# Plotting the three sensors' SWaT data in its first stage

def plot (): 
    plt.figure(figsize=(5,3), dpi=350) 
    df1['LIT101'].plot(label='LIT101', color='orange') 
    df1['AIT201'].plot(label='AIT201')
    df1['LIT301'].plot(label='LIT301') 
    plt.title('First Stage SWaT') 
    # adding Label to the x-axis 
    plt.xlabel('Timestamp')  
    # adding legend to the curve 
    plt.legend(title='Sensors')
    plt.savefig('firststageSwat.png') 
print (plot ()) 


# #### Exploratory Data Analysis

# In[16]:


#If there are missing entries, drop them.
df1.dropna(inplace=True)#,axis=1)  
# Total number of rows and columns 
df1.shape 


# In[17]:


# Dropping the duplicates 
df1= df1.drop_duplicates()
df1.head(2)   


# In[18]:


# Counting the number of rows after removing duplicates.
df1.shape 


# In[1048]:


# Finding the relations between the variables using the correlation 
plt.figure(figsize=(18,10))
c= df1.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
#c  


# In[19]:


# looking the distribution of the data between attack and normal

print(df1.groupby('A')['A'].count()) 


# #### Visualize the dataset 
# -  Plotting the number of normal and Attack transactions in the dataset.

# In[20]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['A'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['A'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations"); 


# In[21]:


# count the number of anomalies and normal data points in our dataset 
df1['A'].value_counts()


# In[22]:


# drop the time stamp cols 
#df1= df1.drop('Timestamp', axis=1)


# #### Normalize using min Max scaler 

# In[23]:


# build the scaler model 
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601'] 
scaler = MinMaxScaler() 
df1[con_feats] = scaler.fit_transform(df1[con_feats])
df1.head() 


# #### Split the Data to train and Test

# In[24]:


# split the normal data with respect to test and Train 
from sklearn.model_selection import train_test_split 
x_good_train, x_good_test = train_test_split(df1, test_size=0.2, random_state=42)   


# In[25]:


# min max scale the input data or Standard Scaler  
x_good_train = x_good_train[x_good_train.A == 0] #where normal transactions 
x_good_train = x_good_train.drop(['A'], axis=1) #drop the class columns 

test_y = x_good_test['A'] # save the class column for the test set 
x_good_test = x_good_test.drop(['A'], axis=1) #drop the class column 

#transform to ndarray both train and testing 
x_good_train = x_good_train.values #transform to ndarray 
x_good_test = x_good_test.values 
x_good_train.shape, x_good_test.shape#,x_good_train.shape,test_y.shape   


# ### Hyperparamter Tuning 

# #### import important libraries 

# In[26]: 


# In[ ]:




