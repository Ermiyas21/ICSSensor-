#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 


# In[9]:


# import data set from the local driver 

# df1=pd.read_csv('train1.csv')
#df1=pd.read_csv('train1_20.csv')
df1=pd.read_csv('train1_20.csv', sep=';' )# 09 month 
df2=pd.read_csv('train2_20.csv', sep=';' ) # 11 month 
df3=pd.read_csv('test1_20.csv', sep=';' ) # 10 month 
df4=pd.read_csv('test2_20.csv', sep=';' ) # 11 month 

frames = [df1,df2,df3,df4] 
df_concat=pd.concat(frames) 
print("The dataset has {} records".format(len(df_concat)))
df_concat.head(5)

# df1.head() 


# In[11]:


df_concat.shape


# In[12]:


# drop the columns that are under attack in each stage. 
df1=df_concat.drop(['attack_P1','attack_P2','attack_P3'],axis=1)#, inplace=True)  


# In[13]:


# a Timestamp as index 
df1= df1.set_index('time') 
df1.head() 


# In[17]:


# Visualizing The first stage with respect to time series  
def plot (): 
    plt.figure(figsize=(16, 8), dpi=150) 
    df1['P1_B3005'].plot(label='P1_B3005', color='orange') 
    df1['P1_B4005'].plot(label='P1_B4005')
    df1['P1_B3004'].plot(label='P1_B3004') 
    # adding title to the plot 
    plt.title('First Stage HAI 2020') 
    # adding Label to the x-axis 
    plt.xlabel('Timestamp') 
    plt.ylabel('data density')
    # adding legend to the curve 
    plt.legend(title='Sensors')
    
print (plot ()) 


# #### Visualize the dataset 
# -  Plotting the number of normal and Attack transactions in the dataset.

# In[18]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['attack'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['attack'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");  


# In[19]:


#Count 1 unique values in each columns
df2=df1.nunique().reset_index()
df2.columns = ['feature','number Of unique']
df2[df2['number Of unique']==1] 


# In[20]:


# drop the time stamp cols 
#df1= df1.drop('P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS', axis=0) 

# drop two columns name is 'C' and 'D'
df1.drop(['P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS'], axis=1)


# In[22]:


import seaborn as sns


#get correlations of each features in dataset
c = df1.corr()
top_corr_features = c.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")  


# In[23]:


c=c.attack.sort_values(ascending=False).head(42).keys().drop('attack') # corelation
print(c)


# In[24]:


# Select Multiple Columns
df2 = df1.loc[:, ['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02D', 'P1_FT01', 'P1_PCV02Z',
       'P1_PCV01Z', 'P1_PCV01D', 'P1_FCV02Z', 'P1_B2016', 'P1_FCV02D',
       'P1_B2004', 'P1_FT01Z', 'P1_B3005', 'P1_PIT01', 'P4_ST_PT01',
       'P2_VYT02', 'P3_LCP01D', 'P2_VXT03', 'P3_LCV01D', 'P1_B3004',
       'P2_VXT02', 'P1_FT03', 'P1_LCV01Z', 'P1_LCV01D', 'P4_HT_LD', 'P1_FT03Z',
       'P4_HT_PO', 'P4_ST_FD', 'P4_LD', 'P4_HT_FD', 'P4_ST_LD', 'P4_ST_PO',
       'P2_24Vdc', 'P2_VYT03', 'P1_TIT02', 'P1_PIT02', 'P2_VT01e', 'P1_FT02',
       'P1_FCV01Z', 'P1_FCV01D', 'P1_B400B','attack']]


# In[25]:


df2.shape


# In[26]:


#To see how the data is spread betwen Attack and Normal 
#print(df1.groupby('Normal/Attack')['Normal/Attack'].count()) 
print(df2.groupby('attack')['attack'].count()) 


# In[27]:


df2.info()


# #### Normalize using min Max scaler 

# In[28]:


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


# #### Split the Data to train and Test 

# In[29]:


# split the normal data with respect to test and Train 
from sklearn.model_selection import train_test_split 
x_good_train, x_good_test = train_test_split(df2, test_size=0.2, random_state=42)   


# In[30]:


# min max scale the input data or Standard Scaler  
x_good_train = x_good_train[x_good_train.attack == 0] #where normal transactions 
x_good_train = x_good_train.drop(['attack'], axis=1) #drop the class columns 

test_y = x_good_test['attack'] # save the class column for the test set 
x_good_test = x_good_test.drop(['attack'], axis=1) #drop the class column 

#transform to ndarray both train and testing 
x_good_train = x_good_train.values #transform to ndarray 
x_good_test = x_good_test.values 
x_good_train.shape, x_good_test.shape#,x_good_train.shape,test_y.shape   

