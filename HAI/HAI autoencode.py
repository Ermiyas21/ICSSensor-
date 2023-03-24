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


# #### import important libraries 

# In[31]:


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


# In[32]:


model = Sequential()
# encoder 
model.add(Dense(128, input_dim=x_good_train.shape[1], activation='relu')) # Input layers 
Dropout(0.01), 
# hidden layers
model.add(Dense(32, activation='relu'))
Dropout(0.01), 
#model.add(Dense(32, activation='relu'))
#Dropout(0.01), 
#Decoder 
model.add(Dense(128, activation='relu')) ## decoder 
Dropout(0.01), 
model.add(Dense(x_good_train.shape[1])) # output layers 
model.compile(loss='msle',metrics=['accuracy'],optimizer='adam')  
model.summary() 


# In[33]:


x_good_test.shape[1] 


# In[34]:


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


# #### Plot training and test loss

# In[35]:


plt.plot(grid.history['loss'])
plt.plot(grid.history['val_loss'])
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

# In[36]:


# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions_HAI = model.predict(x_good_test)#,verbose=1)
mse = np.mean(np.power(x_good_test - test_x_predictions_HAI, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,'attack': test_y}, index=test_y.index)  
error_df.head()


# In[67]:





# In[72]:


#Print memory usage 
model_memory=pd.DataFrame(test_x_predictions_HAI)

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(model_memory) 


# In[38]:


# find the maximum of each column using reconstruction error 
maxValues = error_df.max()
 
print(maxValues) 


# #### Plot Reconsrruction using Error Histogram 
# - to determine the exact point using kernel density estimation method 

# In[39]:


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
#reconstructions = model.predict(normal_test_data)
train_loss = tf.keras.losses.mse(test_x_predictions_HAI, x_good_test)


trainloss=pd.DataFrame(train_loss)
#trainloss.describe()
#trainloss.to_csv(r'/home/jovyan/trainloss.csv',index=False) 
trainloss.describe()   


# In[58]:


# rules-of-thumb to identify the number of bins Freedman–Diaconis rule 
trainloss=pd.DataFrame(trainloss) 
q1 = trainloss.quantile(0.020087)
q3 = trainloss.quantile(0.299992)
iqr = q3 - q1
bin_width = (2 * iqr) / (len(trainloss) ** (1 / 3))
bin_count = int(np.ceil((trainloss.max() - trainloss.min()) / bin_width))
fig = plt.figure(figsize=(7,5.5))
plt.hist(train_loss, bins = bin_count)  
plt.axvline(0.19,0, 3000,color='red', linestyle='dashed', linewidth=1) 
plt.xlabel('RMSE loss ')
plt.ylabel('Density')
plt.title(f'bins - loss distribution = {bin_count}')  


# In[42]:


# finding the number of anomalies using highest reconstruction Error 

outliers = error_df.index[error_df.Reconstruction_error > 0.19].tolist()  
number_of_outliers = len(outliers) 
print("Number of elements in the anomalies: ", number_of_outliers)   


# ### Model Interpretability
# - find the number of anomaly points according to stattical method i.e usinG RMSE loss and bin values  

# In[43]:


# change X_tes_scaled to pandas dataframe
data_n = pd.DataFrame(x_good_test, index= test_y.index)#, columns=numerical_cols) 


# In[44]:


# 
def compute_error_per_dim(point):
    
    initial_pt = np.array(data_n.loc[point,:]).reshape(1,9)
    reconstrcuted_pt = model.predict(initial_pt)
    
    return abs(np.array(initial_pt - reconstrcuted_pt)[0]) 


# In[45]:


# finding the number of anomalies using reconstruction Error 
outliers = error_df.index[error_df.Reconstruction_error >0.19].tolist()  
number_of_outliers = len(outliers) 
print("Number of elements in the anomalies: ", number_of_outliers)  


# #### Calculate RMSE and MAE stastical method  

# In[47]:


# Define a function to calculate MAE and RMSE
errors = test_x_predictions_HAI - x_good_test
mse = np.square(errors).mean()
rmse = np.sqrt(mse)
mae = np.abs(errors).mean()

print('The performance  of autoencoder'+ ':') 
print('')
print('Mean Absolute Error: {:.4f}'.format(mae)) 
print('Mean Square Error:{:.4f}' .format(mse))
print('Root Mean Square Error: {:.4f}'.format(rmse))
print('')    


# #### Classfication report using Stastical method  

# In[56]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.19
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
error_df['pred'] =pred_y
conf_matrix = confusion_matrix(error_df.attack, pred_y)
plt.figure(figsize=(4, 4))

# print confustion matrix 
print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(error_df['attack'], error_df['pred']))
print(" Recall: ",recall_score(error_df['attack'], error_df['pred']))
print(" Precision: ",precision_score(error_df['attack'], error_df['pred']))  
print(classification_report(error_df['attack'], error_df['pred']))   


# #### Visualizing anomaly points  

# In[57]:


# visualize the anomaly points in the dataset with 2D 
plt.figure(figsize=(10,6))  
threshold_fixed = 0.19
groups = error_df.groupby('attack')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Anomalies" if name == 1 else "Normal")
#ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for normal and Anomalies data")
plt.ylabel("Reconstruction error")
plt.xlabel("Time") 

plt.show();  


# #### Mean p-Powered Error for auto encoder 
# - To enhance the perfomance of reall and precision  

# In[161]:


# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions_mean = model.predict(x_good_test)#,verbose=1)
mse = np.mean(np.power(x_good_test - test_x_predictions, 3), axis=1)
error_df_mean = pd.DataFrame({'Reconstruction_error': mse,'attack': test_y}, index=test_y.index)  
error_df_mean.describe() 


# #### Plot Reconsrruction Error Histogram 
# - to determine the exact point instead of pick up manually 

# In[162]:


# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
#reconstructions = model.predict(normal_test_data)
#train_loss = tf.keras.losses.mse(test_x_predictions_mean, x_good_test)
train_loss=error_df_mean['attack']

trainloss=pd.DataFrame(train_loss)
#trainloss.describe()
#trainloss.to_csv(r'/home/jovyan/trainloss.csv',index=False) 
trainloss.describe()   


# In[166]:


# rules-of-thumb to identify the number of bins Freedman–Diaconis rule 
trainloss=pd.DataFrame(trainloss) 
q1 = trainloss.quantile(0.01)
q3 = trainloss.quantile(1)
iqr = q3 - q1
bin_width = (2 * iqr) / (len(trainloss) ** (1 / 3))
bin_count = int(np.ceil((trainloss.max() - trainloss.min()) / bin_width))
fig = plt.figure(figsize=(7,7))
plt.hist(train_loss, bins = bin_count)  
#sns.histplot(x=trainloss,bins=bin_count)
plt.xlabel('MAE loss ')
plt.ylabel('Density')
plt.title(f'bins - loss distribution = {bin_count}')   


# #### Classfication report using Mean p-Powered Error 

# In[179]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.67
pred_y = [1 if e > threshold_fixed else 0 for e in error_df_mean.Reconstruction_error.values]
error_df_mean['pred'] =pred_y
conf_matrix = confusion_matrix(error_df_mean.attack, pred_y)
plt.figure(figsize=(4, 4))

# print confustion matrix 
print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(error_df_mean['attack'], error_df_mean['pred']))
#print(" Recall: ",recall_score(error_df_mean['A'], error_df_mean['pred']))
#print(" Precision: ",precision_score(error_df_mean['A'], error_df_mean['pred']))  
print(classification_report(error_df_mean['attack'], error_df_mean['pred']))    


# In[273]:


# find the maximum of each column
maxValues_mean = error_df_mean.max()
 
print(maxValues_mean)  


# In[274]:


# finding the number of anomalies using reconstruction Error 
outliers_mean = error_df_mean.index[error_df_mean.Reconstruction_error > 0.67].tolist()  
number_of_outliers_mean = len(outliers_mean) 
print("Number of elements in the anomalies: ", number_of_outliers_mean )    


# ### Dynamic way of identifyig threshold  

# #### 1. Logistic regression 
# 
# - Define threshold for our model using logistic regression  

# In[275]:


# Fit logistic regression model on training set 
from sklearn.linear_model import LogisticRegression  


# In[277]:


df_new_HAI = error_df[['Reconstruction_error', 'attack']]
df_new_HAI.head()


# In[279]:


# Plot logistic regression curve
x= df_new_HAI['Reconstruction_error']
y= df_new_HAI['attack']
sns.regplot(x=x, y=y, data=df_new_HAI, logistic=True, ci=None,scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

#pl 
plt.xlabel('Reconstruction_error')
plt.ylabel('Prob of anomalies')
plt.title('Reconstruction_error vs. Prob of anomalies')

plt.show()   


# In[282]:


# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(df_new_HAI.Reconstruction_error, df_new_HAI.attack, test_size=0.2, random_state=321)  


# In[283]:


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


# In[284]:


x_train=x = np.array(x_train)
y_train = np.array(y_train) 


# In[285]:


from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix 
lrt = LogisticRegressionWithThreshold()
lrt.fit(x_train.reshape(-1, 1), y_train)

threshold, optimal_tpr_minus_fpr = lrt.threshold_from_optimal_tpr_minus_fpr(x_train.reshape(-1, 1), y_train)
y_pred = lrt.predict(x_train.reshape(-1, 1), threshold)

threshold, optimal_tpr_minus_fpr   


# In[291]:


from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score 
threshold_fixed = 0.11
pred_y_logs = [1 if e > threshold_fixed else 0 for e in df_new_HAI.Reconstruction_error.values]
df_new_HAI['pred'] =pred_y_logs
conf_matrix = confusion_matrix(df_new_HAI.attack, pred_y)
plt.figure(figsize=(4, 4))

print(conf_matrix)
# print Accuracy, precision and recall
print(" Accuracy: ",accuracy_score(df_new_HAI['attack'], df_new_HAI['pred']))
#print(" Recall: ",recall_score(df_new['A'], df_new['pred']))
#print(" Precision: ",precision_score(df_new['A'], df_new['pred']))  
print(classification_report(df_new_HAI['attack'], df_new_HAI['pred']))     


# In[ ]:




