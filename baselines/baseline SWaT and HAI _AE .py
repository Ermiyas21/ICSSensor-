#!/usr/bin/env python
# coding: utf-8

# #### Import required libraries 

# In[146]:


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


# In[147]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV  


# #### Read the dataset 
# - data normal 

# In[148]:


# # import data set from the local driver 
# dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
# dff.head() 

# import data set from the local driver 
dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
dff.head() 
df=pd.read_csv('SWaT_Dataset_Attack_v0 - Copy.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp')
frames = [dff,df] 
df_concat=pd.concat(frames) 
df_concat.head(5)     


# In[149]:


# # convert timestamp to DateTime 
df_concat['Timestamp'] = pd.to_datetime(df_concat['Timestamp']) 
df_concat.head()


# In[150]:


# remove the space on Normal/Attack columns 
df_concat['Normal/Attack'] = df_concat['Normal/Attack'].str.replace(' ', '')  
#To see how the data is spread betwen Attack and Normal 
print(df_concat.groupby('Normal/Attack')['Normal/Attack'].count())  
# Rename the col name Normal/Attack with A   
df_concat.rename(columns = {'Normal/Attack':'A'}, inplace = True)
df_concat.head(2)  


# In[152]:


# Convert non-numeric class to numeric

df_concat.A[df_concat.A== 'Normal'] = 0 
df_concat.A[df_concat.A == 'Attack'] = 1    
df_concat.head()


# In[153]:


# make the class as float 
df_concat['A'] = df_concat['A'].astype('float')  


# In[154]:


# a Timestamp as index 
dff= df_concat.set_index('Timestamp') 
dff.head() 


# In[155]:


dff.shape 


# In[156]:


# Plotting the three sensors' SWaT data in its first stage

def plot (): 
    plt.figure(figsize=(16, 8), dpi=150) 
    dff['LIT101'].plot(label='LIT101', color='orange') 
    dff['AIT201'].plot(label='AIT201')
    dff['LIT301'].plot(label='LIT301') 
    plt.title('First Stage SWaT') 
    # adding Label to the x-axis 
    plt.xlabel('Timestamp') 
    # adding legend to the curve 
    plt.legend(title='Sensors')
    
print (plot ()) 


# #### Exploratory Data Analysis

# In[157]:


#If there are missing entries, drop them.
dff.dropna(inplace=True)#,axis=1)  
# Total number of rows and columns 
dff.shape 


# In[158]:


# Dropping the duplicates 
dff= dff.drop_duplicates()
dff.head(2)   


# In[159]:


# Counting the number of rows after removing duplicates.
dff.shape 
 


# In[160]:


# looking the distribution of the data between attack and normal

print(dff.groupby('A')['A'].count())  


# #### Visualize the dataset 
# -  Plotting the number of normal and Attack transactions in the dataset. 

# In[161]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(dff['A'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(dff['A'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class") 
plt.ylabel("Number of Observations");  


# In[162]:


dff.info()


# #### Normalize using min Max scaler  

# In[163]:


# build the scaler model 
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601'] 
scaler = MinMaxScaler() 
dff[con_feats] = scaler.fit_transform(dff[con_feats])
dff.head()  


# #### Split the Data to train and Test 

# In[164]:


# split the normal data with respect to test and Train 
from sklearn.model_selection import train_test_split 
x_good_train, x_good_test = train_test_split(dff, test_size=0.2, random_state=42)    


# In[165]:


# min max scale the input data or Standard Scaler  
x_good_train = x_good_train[x_good_train.A == 0] #where normal transactions 
x_good_train = x_good_train.drop(['A'], axis=1) #drop the class columns 

test_y = x_good_test['A'] # save the class column for the test set 
x_good_test = x_good_test.drop(['A'], axis=1) #drop the class column 

#transform to ndarray both train and testing 
x_good_train = x_good_train.values #transform to ndarray 
x_good_test = x_good_test.values 
x_good_train.shape, x_good_test.shape#,x_good_train.shape,test_y.shape    


# #### import important libraries  

# In[166]:


#Define the autoencoder model
from sklearn import metrics
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  
from keras.layers import Input, Dense
from keras import regularizers 
from keras.models import Model, load_model 
import datetime  


# ##### Build model  

# In[167]:


model = Sequential()
# encoder 
model.add(Dense(128, input_dim=x_good_train.shape[1], activation='relu')) # Input layers 
Dropout(0.01), 
# hidden layers
model.add(Dense(32, activation='relu'))
Dropout(0.01), 
#Decoder 
model.add(Dense(128, activation='relu')) ## decoder 
Dropout(0.01), 
model.add(Dense(x_good_train.shape[1])) # output layers 
model.compile(loss='msle',metrics=['accuracy'],optimizer='adam')  
model.summary()  


# In[168]:


# chcking the diminsion
x_good_test.shape[1]  


# In[169]:


import time  


# In[170]:



t0 = time.time() 
#model.fit(x_good_train,x_good_train,verbose=1,epochs=100) 
grid=model.fit(
    x_good_train,x_good_train,
    verbose=2,
    epochs=15,
    batch_size=256,
    validation_data=(x_good_test, x_good_test), 
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ])  
score=model.evaluate(x_good_test, x_good_test, verbose=1)
print('Test loss:', score[0])  
print('Accuracy:', score[1])  
# print Model
print("Training time:", time.time()-t0)  


# #### Plot training and test loss 

# In[171]:


plt.plot(grid.history['loss'])
plt.plot(grid.history['val_loss'])
#plt.plot(grid.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.legend(['loss','val_loss'])#,'accuracy'])
plt.show()  


# ### Detect Anomalies on test data  
# - Data points with higher reconstruction loss are considered anomalies. 
# - To calculate the reconstruction loss on test data, predict the test data and calculate the root mean square error between the test data and the reconstructed test data. 
# 
# #### 1. Predictions and Computing Reconstruction Error using RMSE   

# In[172]:


# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions = model.predict(x_good_test)#,verbose=1)
mse = np.mean(np.power(x_good_test - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,'A': test_y}, index=test_y.index)  
error_df.head() 


# In[231]:


#Print memory usage 
model_memory=pd.DataFrame(test_x_predictions)

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(model_memory)  


# In[173]:


# find the maximum RMSE values using reconstruction error 
maxValues = error_df.max()
 
print(maxValues)  


# In[174]:


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
#reconstructions = model.predict(normal_test_data)
train_loss1 = tf.keras.losses.mse(test_x_predictions, x_good_test)


trainloss1=pd.DataFrame(train_loss1)
#trainloss.describe()
#trainloss.to_csv(r'/home/jovyan/trainloss.csv',index=False) 
trainloss1.describe()  
 


# In[185]:


# rules-of-thumb to identify the number of bins Freedman–Diaconis rule 
trainloss1=pd.DataFrame(trainloss1) 
q1 = trainloss1.quantile(0)
q3 = trainloss1.quantile(1)
iqr = q3 - q1
bin_width = (2 * iqr) / (len(trainloss1) ** (1 / 3))
bin_count = int(np.ceil((trainloss1.max() - trainloss1.min()) / bin_width))
fig = plt.figure(figsize=(7,5.5))
plt.hist(train_loss1, bins = bin_count)  

plt.axvline(0.33,0, 9000,color='red', linestyle='dashed', linewidth=1) 
plt.xlabel('RMSE loss ')
plt.ylabel('Density')
plt.title(f'bins - loss distribution = {bin_count}') 


# In[186]:


# finding the number of anomalies using highest reconstruction Error 
outliers = error_df.index[error_df.Reconstruction_error >0.33].tolist()  
number_of_outliers = len(outliers) 
print("Number of elements in the anomalies: ", number_of_outliers)  


# #### Calculate RMSE and MAE stastical method  

# In[187]:


# Define a function to calculate MAE and RMSE
errors = test_x_predictions - x_good_test
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors).mean()

print('The performance  of autoencoder'+ ':') 
print('')
print('Mean Absolute Error: {:.4f}'.format(mae)) 
print('Mean Square Error:{:.4f}' .format(mse))
print('Root Mean Square Error: {:.4f}'.format(rmse))
print('')   


# In[188]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.33
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
error_df['pred'] =pred_y
conf_matrix = confusion_matrix(error_df.A, pred_y)
plt.figure(figsize=(4, 4))

print(conf_matrix)

# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(error_df['A'], error_df['pred']))
print(classification_report(error_df['A'], error_df['pred']))  


# #### For HAI data set 

# In[189]:


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


# In[190]:


df_concat.shape 


# In[191]:


# drop the columns that are under attack in each stage. 
df1=df_concat.drop(['attack_P1','attack_P2','attack_P3'],axis=1)#, inplace=True)   


# In[192]:


# a Timestamp as index 
df1= df1.set_index('time') 
df1.head()  


# In[193]:


# Visualizing The first stage with respect to time series  
def plot (): 
    plt.figure(figsize=(16, 8), dpi=150) 
    df1['P1_B3005'].plot(label='P1_B3005', color='orange') 
    df1['P1_B4005'].plot(label='P1_B4005')
    df1['P1_B3004'].plot(label='P1_B3004') 
    # adding title to the plot 
    plt.title('First Stage HAI') 
    # adding Label to the x-axis 
    plt.xlabel('Timestamp') 
    # adding legend to the curve 
    plt.legend(title='Sensors')
    
print (plot ())  


# In[194]:


# looking the distribution of the data between attack and normal

print(df1.groupby('attack')['attack'].count())   


# #### Visualize the dataset 
# -  Plotting the number of normal and Attack transactions in the dataset. 

# In[195]:


#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['attack'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['attack'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");   


# In[196]:


#Count 1 unique values in each columns
df2=df1.nunique().reset_index()
df2.columns = ['feature','number Of unique']
df2[df2['number Of unique']==1] 


# In[197]:


# drop the time stamp cols 
#df1= df1.drop('P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS', axis=0) 

# drop two columns name is 'C' and 'D'
df1.drop(['P2_Auto','P2_Emgy','P2_On','P2_TripEx','P3_LH','P3_LL','P4_HT_PS'], axis=1) 


# In[198]:


import seaborn as sns


#get correlations of each features in dataset
c = df1.corr()
top_corr_features = c.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")   


# In[199]:


c=c.attack.sort_values(ascending=False).head(42).keys().drop('attack') # corelation
print(c)


# In[ ]:





# In[200]:


# Select Multiple Columns
df2 = df1.loc[:, ['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02Z', 'P1_B4002', 'P1_FT01',
       'P1_FT01Z', 'P1_B2004', 'P3_LCP01D', 'P1_FT03', 'P1_FT03Z',
       'P1_B3005', 'P1_FCV01D', 'P1_FCV01Z', 'P1_B4022', 'P1_LCV01D',
       'P1_FT02', 'P1_LCV01Z', 'P1_B4005', 'P1_FCV02Z', 'P1_B400B',
       'P1_FT02Z', 'P2_VYT02', 'P1_FCV02D', 'P4_ST_FD', 'P1_PCV01Z', 'P4_HT_FD',
       'P2_24Vdc', 'P1_PCV01D', 'P2_VXT03', 'P2_VXT02', 'P2_VT01e', 'P2_VYT03',
       'P4_HT_LD', 'P1_PIT01', 'P4_HT_PO', 'P3_LCV01D', 'P3_LT01', 'P4_ST_TT01',
       'P4_ST_PT01', 'P4_LD', 'P4_ST_LD','attack']]


# In[203]:


#To see how the data is spread betwen Attack and Normal 
#print(df1.groupby('Normal/Attack')['Normal/Attack'].count()) 
print(df2.groupby('attack')['attack'].count())  


# #### Normalize using min Max scaler  

# In[204]:


con_feats = ['P1_FCV03D', 'P1_FCV03Z', 'P1_PCV02Z', 'P1_B4002', 'P1_FT01',
       'P1_FT01Z', 'P1_B2004', 'P3_LCP01D', 'P1_FT03', 'P1_FT03Z',
       'P1_B3005', 'P1_FCV01D', 'P1_FCV01Z', 'P1_B4022', 'P1_LCV01D',
       'P1_FT02', 'P1_LCV01Z', 'P1_B4005', 'P1_FCV02Z', 'P1_B400B',
       'P1_FT02Z', 'P2_VYT02', 'P1_FCV02D', 'P4_ST_FD', 'P1_PCV01Z', 'P4_HT_FD',
       'P2_24Vdc', 'P1_PCV01D', 'P2_VXT03', 'P2_VXT02', 'P2_VT01e', 'P2_VYT03',
       'P4_HT_LD', 'P1_PIT01', 'P4_HT_PO', 'P3_LCV01D', 'P3_LT01', 'P4_ST_TT01',
       'P4_ST_PT01', 'P4_LD', 'P4_ST_LD'] 


scaler = MinMaxScaler() 
df2[con_feats] = scaler.fit_transform(df2[con_feats])
df2.head()   


# #### Split the Data to train and Test  

# In[210]:


# split the normal data with respect to test and Train 
from sklearn.model_selection import train_test_split 
x_good_train_HAI, x_good_test_HAI = train_test_split(df1, test_size=0.2, random_state=42)   


# In[211]:


# min max scale the input data or Standard Scaler  
x_good_train_HAI = x_good_train_HAI[x_good_train_HAI.attack == 0] #where normal transactions 
x_good_train_HAI = x_good_train_HAI.drop(['attack'], axis=1) #drop the class columns 

test_y = x_good_test_HAI['attack'] # save the class column for the test set 
x_good_test_HAI = x_good_test_HAI.drop(['attack'], axis=1) #drop the class column 

#transform to ndarray both train and testing 
x_good_train_HAI = x_good_train_HAI.values #transform to ndarray 
x_good_test_HAI = x_good_test_HAI.values 
x_good_train_HAI.shape, x_good_test_HAI.shape#,x_good_train.shape,test_y.shape   


# #### import important libraries  

# In[133]:


######### 
#Define the autoencoder model
#Since we're dealing with numeric values we can use only Dense layers.

from sklearn import metrics
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  
from keras.layers import Input, Dense
from keras import regularizers 
from keras.models import Model, load_model 
import datetime  


# In[212]:


model2 = Sequential()
# encoder 
model2.add(Dense(128, input_dim=x_good_train_HAI.shape[1], activation='relu')) # Input layers 
Dropout(0.01), 
# hidden layers
model2.add(Dense(32, activation='relu'))
Dropout(0.01), 
#model.add(Dense(32, activation='relu'))
#Dropout(0.01), 
#Decoder 
model2.add(Dense(128, activation='relu')) ## decoder 
Dropout(0.01), 
model2.add(Dense(x_good_train_HAI.shape[1])) # output layers 
model2.compile(loss='msle',metrics=['accuracy'],optimizer='adam')  
model2.summary()  


# In[214]:


#model.fit(x_good_train,x_good_train,verbose=1,epochs=100) 
grid2=model2.fit(
    x_good_train_HAI,x_good_train_HAI,
    verbose=2,
    epochs=15,
    batch_size=256,
    validation_data=(x_good_test_HAI, x_good_test_HAI), 
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ])  
score1=model2.evaluate(x_good_test_HAI, x_good_test_HAI, verbose=1)
print('Test loss:', score1[0]) 
print('Accuracy:', score1[1])   


# #### Plot training and test loss 

# In[215]:


plt.plot(grid2.history['loss'])
plt.plot(grid2.history['val_loss'])
#plt.plot(grid.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss','val_loss'])#,'accuracy'])
plt.show()   


# ### Detect Anomalies on test data  
# - Anomalies are data points where the reconstruction loss is higher 
# - To calculate the reconstruction loss on test data, predict the test data and calculate the mean square error between the test data and the reconstructed test data. 
# 
# #### 1. Predictions and Computing Reconstruction Error using RMSE  

# In[216]:


# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions_HAI = model2.predict(x_good_test_HAI)#,verbose=1)
mse = np.mean(np.power(x_good_test_HAI - test_x_predictions_HAI, 2), axis=1)
error_df_HAI = pd.DataFrame({'Reconstruction_error': mse,'attack': test_y}, index=test_y.index)  
error_df_HAI.head() 


# In[217]:


# find the maximum of each column using reconstruction error 
maxValues_HAI = error_df_HAI.max()
 
print(maxValues_HAI)  


# In[218]:


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
#reconstructions = model.predict(normal_test_data)
train_loss2 = tf.keras.losses.mse(test_x_predictions_HAI, x_good_test_HAI)


trainloss_HAI=pd.DataFrame(train_loss2)
#trainloss.describe()
#trainloss.to_csv(r'/home/jovyan/trainloss.csv',index=False) 
trainloss_HAI.describe()  
  


# In[219]:


# rules-of-thumb to identify the number of bins Freedman–Diaconis rule 
trainloss_HAI=pd.DataFrame(trainloss_HAI) 
q1 = trainloss_HAI.quantile(0)
q3 = trainloss_HAI.quantile(1)
iqr = q3 - q1
bin_width = (2 * iqr) / (len(trainloss_HAI) ** (1 / 3))
bin_count = int(np.ceil((trainloss_HAI.max() - trainloss_HAI.min()) / bin_width))
fig = plt.figure(figsize=(7,5.5))
plt.hist(train_loss1, bins = bin_count)  

plt.axvline(0.53,0, 9000,color='red', linestyle='dashed', linewidth=1) 
plt.xlabel('RMSE loss ')
plt.ylabel('Density')
plt.title(f'bins - loss distribution = {bin_count}') 


# In[224]:


# finding the number of anomalies using highest reconstruction Error 

outliers = error_df.index[error_df.Reconstruction_error > 0.4].tolist()  
number_of_outliers = len(outliers) 
print("Number of elements in the anomalies: ", number_of_outliers)    


# #### Calculate RMSE and MAE stastical method 

# In[225]:


# Define a function to calculate MAE and RMSE
errors_HAI = test_x_predictions_HAI - x_good_test_HAI
mse = np.square(errors_HAI).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors_HAI).mean()

print('The performance  of autoencoder'+ ':') 
print('')
print('Mean Absolute Error: {:.4f}'.format(mae)) 
print('Mean Square Error:{:.4f}' .format(mse))
print('Root Mean Square Error: {:.4f}'.format(rmse))
print('')     


# In[229]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.7
pred_y = [1 if e > threshold_fixed else 0 for e in error_df_HAI.Reconstruction_error.values]
error_df_HAI['pred'] =pred_y
conf_matrix = confusion_matrix(error_df_HAI.attack, pred_y)
plt.figure(figsize=(4, 4))

print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(error_df_HAI['attack'], error_df_HAI['pred']))
print(classification_report(error_df_HAI['attack'], error_df_HAI['pred']))   


# In[ ]:




