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


# In[27]:


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


# In[28]:


x_good_test.shape[1] 


# In[29]:


import time 


# - Define the callbacks for checkpoints and early stopping 
# - Compile the Autoencoder 
# - Train the Autoencoder 

# In[30]:



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

# In[31]:


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

# In[32]:


# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions = model.predict(x_good_test)#,verbose=1)
mse = np.mean(np.power(x_good_test - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,'A': test_y}, index=test_y.index)  
error_df.head()


# In[207]:


#Print memory usage 
model_memory=pd.DataFrame(test_x_predictions)

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(model_memory) 


# In[33]:


# find the maximum RMSE values using reconstruction error 
maxValues = error_df.max()
 
print(maxValues) 


# #### Plot Reconsrruction using Error Histogram 
# - To determine the exact point using kernel density estimation method 

# In[34]:


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
train_loss = tf.keras.losses.mse(test_x_predictions, x_good_test)
trainloss=pd.DataFrame(train_loss)
trainloss.describe()  
 


# In[47]:


# rules-of-thumb to identify the number of bins Freedman–Diaconis rule 
trainloss=pd.DataFrame(trainloss) 
q1 = trainloss.quantile(0.021579)
q3 = trainloss.quantile(0.466866)
iqr = q3 - q1
bin_width = (2 * iqr) / (len(trainloss) ** (1 / 3))
bin_count = int(np.ceil((trainloss.max() - trainloss.min()) / bin_width))
fig = plt.figure(figsize=(7,5.5))
plt.hist(train_loss, bins = bin_count) 
plt.axvline(0.35,0, 3000,color='red', linestyle='dashed', linewidth=1)
#sns.histplot(x=trainloss,bins=bin_count)
plt.xlabel('RMSE loss ')
plt.ylabel('Density')
plt.title(f'bins - loss distribution = {bin_count}') 


# ### Model Interpretability
# - find the number of anomaly points according to stattical method i.e usinG RMSE loss and bin values 

# In[1066]:


# change X_tes_scaled to pandas dataframe
data_n = pd.DataFrame(x_good_test, index= test_y.index)#, columns=numerical_cols) 


# In[1067]:


def compute_error_per_dim(point):
    
    initial_pt = np.array(data_n.loc[point,:]).reshape(1,9)
    reconstrcuted_pt = model.predict(initial_pt)
    
    return abs(np.array(initial_pt - reconstrcuted_pt)[0]) 


# In[37]:


# finding the number of anomalies using reconstruction Error 
outliers = error_df.index[error_df.Reconstruction_error >0.35].tolist()  
number_of_outliers = len(outliers) 
print("Number of elements in the anomalies: ", number_of_outliers)  


# #### Calculate RMSE and MAE stastical method 

# In[216]:


# Define a function to calculate MAE and RMSE
errors = test_x_predictions - x_good_test
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors).mean()

print('The performance  of autoencoder'+ ':') 
print('')
print('Mean Absolute Error: {:.4f}' .format(mae)) 
print('Mean Square Error:{:.4f}' .format(mse))
print('Root Mean Square Error: {:.4f}' .format(rmse))
print('')   


# #### Classfication report using Stastical method 

# In[225]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.35
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
error_df['pred'] =pred_y
conf_matrix = confusion_matrix(error_df.A, pred_y)
plt.figure(figsize=(4, 4))

# print confustion matrix 
print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(error_df['A'], error_df['pred']))
print(" Recall: ",recall_score(error_df['A'], error_df['pred']))
print(" Precision: ",precision_score(error_df['A'], error_df['pred']))  
print(classification_report(error_df['A'], error_df['pred']))   


# #### Visualizing anomaly points 

# In[48]:


# visualize the anomaly points in the dataset with 2D 
plt.figure(figsize=(10,6))  
threshold_fixed = 0.35
groups = error_df.groupby('A')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Anomalies" if name == 1 else "Normal")
#ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
#ax.hlines(ax.get_xlim()[0], ax.get_xlim()[1])#, colors="r", zorder=100)#, label='Threshold') 
ax.legend(title='data points catagory')
plt.title("normal and anomalies data points for SWaT")
plt.ylabel("Reconstruction error")
plt.xlabel("Time") 
  
#plt.savefig('SWaT_AE_1sed.png') 


# #### Mean p-Powered Error for auto encoder 
# - To enhance the perfomance of reall and precision  

# In[127]:


# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions_mean = model.predict(x_good_test)#,verbose=1)
mse2 = np.mean(np.power(x_good_test - test_x_predictions_mean, 10), axis=1)
error_df_mean = pd.DataFrame({'Reconstruction_error': mse2,'A': test_y}, index=test_y.index)  
error_df_mean.describe()


# In[128]:


# find the maximum of each column
maxValues_mean = error_df_mean.max()
 
print(maxValues_mean) 


# #### Plot Reconsrruction Error Histogram 
# - to determine the exact point instead of pick up manually 

# In[129]:


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# # reconstruction loss for normal test data
# #reconstructions = model.predict(normal_test_data)
# #train_loss = tf.keras.losses.mse(test_x_predictions_mean, x_good_test)
# train_loss=error_df_mean['A']

# trainloss=pd.DataFrame(train_loss)
# #trainloss.describe()
# #trainloss.to_csv(r'/home/jovyan/trainloss.csv',index=False) 
# trainloss.describe()   


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
#train_loss2 = tf.keras.losses.mse(test_x_predictions_mean, x_good_test)
train_loss2=mse2
trainloss2=pd.DataFrame(train_loss2)
trainloss2.describe()   


# In[132]:


# rules-of-thumb to identify the number of bins Freedman–Diaconis rule 
trainloss2=pd.DataFrame(trainloss2) 
q1 = trainloss2.quantile(0.000010)
q3 = trainloss2.quantile(1)
iqr = q3 - q1
bin_width = (2 * iqr) / (len(trainloss2) ** (1 / 3))
bin_count = int(np.ceil((trainloss2.max() - trainloss2.min()) / bin_width))
fig = plt.figure(figsize=(7,7))
#plt.axvline(0.4,0, 3000,color='red', linestyle='dashed', linewidth=1)
plt.hist(train_loss, bins = bin_count)  
#sns.histplot(x=trainloss,bins=bin_count)
plt.xlabel('MAE loss ')
plt.ylabel('Density')
plt.title(f'bins - loss distribution = {bin_count}')  


# In[145]:


# finding the number of anomalies using reconstruction Error 
outliers_mean = error_df_mean.index[error_df_mean.Reconstruction_error > 20.4].tolist()  
number_of_outliers_mean = len(outliers_mean) 
print("Number of elements in the anomalies: ", number_of_outliers_mean ) 


# #### Classfication report using Mean p-Powered Error

# In[143]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 20.4
pred_y = [1 if e > threshold_fixed else 0 for e in error_df_mean.Reconstruction_error.values]
error_df_mean['pred'] =pred_y
conf_matrix = confusion_matrix(error_df_mean.A, pred_y)
plt.figure(figsize=(4, 4))

# print confustion matrix 
print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(error_df_mean['A'], error_df_mean['pred']))
#print(" Recall: ",recall_score(error_df_mean['A'], error_df_mean['pred']))
#print(" Precision: ",precision_score(error_df_mean['A'], error_df_mean['pred']))  
print(classification_report(error_df_mean['A'], error_df_mean['pred']))   


# ### Dynamic way of identifyig threshold 

# #### 1. Logistic regression 
# 
# - Define threshold for our model using logistic regression 

# In[146]:


# Fit logistic regression model on training set 
from sklearn.linear_model import LogisticRegression 


# In[147]:


df_new = error_df[['Reconstruction_error', 'A']]
df_new.head()


# - Reconstruction Error values are shown on the x-axis, and the predicted probability, or anomalies or normal data points, is shown on the y-axis. 
# 
# - It is clear that normal probability are related with larger values of reconstruction error (normal data points) 
# - The figure also included the threshold values, which were either quite close or 0.15. 

# In[163]:


# Plot logistic regression curve
x= df_new['Reconstruction_error']
y= df_new['A']
sns.regplot(x=x, y=y, data=df_new, logistic=True, ci=None,
            scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

#pl 
plt.xlabel('Reconstruction_error')
plt.ylabel('Prob of anomalies')
plt.title('Reconstruction_error vs. Prob of anomalies')

plt.show()  


# In[150]:


# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(df_new.Reconstruction_error, df_new.A, test_size=0.2, 
                                                    random_state=42) 


# In[151]:


class LogisticRegressionWithThreshold(LogisticRegression):
    def predict(self, X, threshold=None):
        if threshold == None: # If no threshold passed in, simply call the base class predict, effectively threshold=0.5
            return LogisticRegression.predict(self, X)
        else:
            y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
            y_pred_with_threshold = (y_scores >= threshold).astype(int)

            return y_pred_with_threshold
    
    def threshold_from_optimal_tpr_minus_fpr(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_scores) 

        optimal_idx = np.argmax(tpr - fpr)

        return thresholds[optimal_idx], tpr[optimal_idx] - fpr[optimal_idx]  


# In[152]:


x_train=x = np.array(x_train)
y_train = np.array(y_train) 


# In[153]:


from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix 
lrt = LogisticRegressionWithThreshold()
lrt.fit(x_train.reshape(-1, 1), y_train)

threshold, optimal_tpr_minus_fpr = lrt.threshold_from_optimal_tpr_minus_fpr(x_train.reshape(-1, 1), y_train)
y_pred = lrt.predict(x_train.reshape(-1, 1), threshold)

threshold, optimal_tpr_minus_fpr  


# In[161]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.63
pred_y_logs = [1 if e > threshold_fixed else 0 for e in df_new.Reconstruction_error.values]
df_new['pred'] =pred_y_logs
conf_matrix = confusion_matrix(df_new.A, pred_y)
plt.figure(figsize=(4, 4))

print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(df_new['A'], df_new['pred']))
#print(" Recall: ",recall_score(df_new['A'], df_new['pred']))
#print(" Precision: ",precision_score(df_new['A'], df_new['pred']))  
print(classification_report(df_new['A'], df_new['pred']))    


# ### 2. CUSUM  
# -  The developed ADS encapsulates a detection algorithm that records the gradual change of the monitored variables, and issues an alert if the deviation is persistent. 
# - In the above visual, the anomaly score peaks at time 100 and time 200, which corresponds to points where massive shifts in the time series occur. Setting a minimum threshold for anomaly scores, where anything above a certain threshold corresponds to a change point in the sequence, is the best way to identify individual change points in the series.

# In[174]:


# We can take the cumsum of the Reconstruction_error column to see how it changes, and 
# create a new column called Cumsum:
error_df['Cumsum'] = error_df['pred'].cumsum()  
error_df.head()  


# In[206]:


# Furthermore, we can look into whether the change is positive or negative: 
error_df['Change'] = np.where(error_df['A'] < error_df['pred'].shift(), '+', '-')
error_df.head(5) 


# In[181]:


# find the maximum of each column using reconstruction error 
maxValues = error_df['Cumsum'].max()
print(maxValues)  


# 

# In[167]:


# We can also investigate whether the change is going in the positive or negative direction: 
error_df['Cumsum'] = error_df[['Reconstruction_error','A']].groupby('A').cumsum()
error_df


# In[189]:


# We can also set an arbitrary threshold and see when the cumsum is above or below it: 

error_df ['Threshold'] = np.where(error_df ['Cumsum'] < 0.4, 'Past Threshold', '+')
error_df.head(2)  


# #### 

# In[266]:


# Minimum and maximum salaries
#print (max(0,error_df ['pred'])) #-df1['FIT101']))
# print('Minimum salary ' + str(salary_df['Salary (in USD)'].min()))
# print('Maximum salary ' + str(salary_df['Salary (in USD)'].max())) 


# In[255]:


#detect_cusum detects changes in a single column x
def detect_cusum(x, threshold=1, drift=0, show=True, ax=None):
    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta = np.array([], dtype=int)
    tap, tan = 0, 0
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change  (so we are computing the cumulative sum of changes(see formula of 's'))
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            #tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    return ta 

#_is_change loops over all the columns and calls the detect_cusum function for each column
def _is_change(df1): 


    keep=['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601']

        #preprocessing
    df = df1[keep].astype(float)
    df=(df-df.mean())/df.std() #standardize
    df=df.fillna(-1) #for a given window size, the standardization sometimes yields Nan because the column is constant. Replace those by -1.
    df['prediction']=0
    #loop over all the columns
    for i in df.columns:
        x = df[i].to_numpy()
        ta=detect_cusum(x,1.5, 0.75, True) 
        for j in ta:
           df['prediction'][j]=1


    #merge the labels back after prediction
    df['A']=df1['A'] 
    df[keep]=df1[keep]#get back unnormalized values 
    return df  


# In[ ]:





# In[ ]:




