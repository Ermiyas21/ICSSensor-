# ! pip install pyanom
# # ! conda install tensorflow -y


```python
# ! conda install  pyanom
```

#### Import required libraries 


```python
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

```


```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
```

#### Read the dataset 
- data integration both normal and attack 


```python
# import data set from the local driver 
dff=pd.read_csv('SWaT_Dataset_Normal_v0.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp') 
df=pd.read_csv('SWaT_Dataset_Attack_v0 - Copy.csv')#, parse_dates = ['Timestamp'], index_col = 'Timestamp')
frames = [dff,df] 
df_concat=pd.concat(frames) 
df_concat.head(5)  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>MV101</th>
      <th>P101</th>
      <th>P102</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>...</th>
      <th>P501</th>
      <th>P502</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>P601</th>
      <th>P602</th>
      <th>P603</th>
      <th>Normal/Attack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22/12/2015 4:00:00 PM</td>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22/12/2015 4:00:01 PM</td>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22/12/2015 4:00:02 PM</td>
      <td>2.439548</td>
      <td>260.9131</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.467305</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22/12/2015 4:00:03 PM</td>
      <td>2.428338</td>
      <td>260.2850</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22/12/2015 4:00:04 PM</td>
      <td>2.424815</td>
      <td>259.8925</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>244.4245</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>




```python
# df_concat.to_csv(r'/home/jovyan/SWaT dataset.csv',index=False) 
```

#### Automatic Sensor data extraction 


```python
# Extract the sensor components that begin with the list 
df1=df_concat.filter(regex='(^Time|^PIT|^AIT|^FIT|^DPI|^LIT|^Norma)',axis=1)#.head()
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>Normal/Attack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22/12/2015 4:00:00 PM</td>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>20.79839</td>
      <td>2.235275</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22/12/2015 4:00:01 PM</td>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>20.79839</td>
      <td>2.234507</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22/12/2015 4:00:02 PM</td>
      <td>2.439548</td>
      <td>260.9131</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.467305</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22/12/2015 4:00:03 PM</td>
      <td>2.428338</td>
      <td>260.2850</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.2799</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22/12/2015 4:00:04 PM</td>
      <td>2.424815</td>
      <td>259.8925</td>
      <td>244.4245</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.1597</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
# # timestamp to DateTime 
df1['Timestamp'] = pd.to_datetime(df1['Timestamp']) 
df1.head()
```

    /tmp/ipykernel_2367236/4044833211.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>Normal/Attack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-12-22 16:00:00</td>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>20.79839</td>
      <td>2.235275</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-22 16:00:01</td>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>20.79839</td>
      <td>2.234507</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-22 16:00:02</td>
      <td>2.439548</td>
      <td>260.9131</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.467305</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-22 16:00:03</td>
      <td>2.428338</td>
      <td>260.2850</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.2799</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-22 16:00:04</td>
      <td>2.424815</td>
      <td>259.8925</td>
      <td>244.4245</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.1597</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
# remove the space on Normal/Attack columns 
df1['Normal/Attack'] = df1['Normal/Attack'].str.replace(' ', '')  
#To see how the data is spread betwen Attack and Normal 
print(df1.groupby('Normal/Attack')['Normal/Attack'].count())  
# Rename the col name Normal/Attack with A   
df1.rename(columns = {'Normal/Attack':'A'}, inplace = True)
df1.head(2)  

```

    Normal/Attack
    Attack     54621
    Normal    892098
    Name: Normal/Attack, dtype: int64


    /tmp/ipykernel_2367236/3530696964.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['Normal/Attack'] = df1['Normal/Attack'].str.replace(' ', '')
    /tmp/ipykernel_2367236/3530696964.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1.rename(columns = {'Normal/Attack':'A'}, inplace = True)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-12-22 16:00:00</td>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>20.79839</td>
      <td>2.235275</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-22 16:00:01</td>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>20.79839</td>
      <td>2.234507</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>




```python
# Convert non-numeric class to numeric

df1.A[df1.A== 'Normal'] = 0 
df1.A[df1.A == 'Attack'] = 1    
df1.head()
```

    /tmp/ipykernel_2367236/3765477871.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1.A[df1.A== 'Normal'] = 0
    /tmp/ipykernel_2367236/3765477871.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1.A[df1.A== 'Normal'] = 0
    /tmp/ipykernel_2367236/3765477871.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1.A[df1.A == 'Attack'] = 1
    /tmp/ipykernel_2367236/3765477871.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1.A[df1.A == 'Attack'] = 1





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-12-22 16:00:00</td>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>20.79839</td>
      <td>2.235275</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-22 16:00:01</td>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>20.79839</td>
      <td>2.234507</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-22 16:00:02</td>
      <td>2.439548</td>
      <td>260.9131</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.467305</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.4401</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-22 16:00:03</td>
      <td>2.428338</td>
      <td>260.2850</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.2799</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-22 16:00:04</td>
      <td>2.424815</td>
      <td>259.8925</td>
      <td>244.4245</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.1597</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
# make the class as float 
df1['A'] = df1['A'].astype('float') 
```

    /tmp/ipykernel_2367236/1868148734.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['A'] = df1['A'].astype('float')



```python
# a Timestamp as index 
df1= df1.set_index('Timestamp') 
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>AIT401</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>A</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-22 16:00:00</th>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>20.79839</td>
      <td>2.235275</td>
      <td>327.4401</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:01</th>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>20.79839</td>
      <td>2.234507</td>
      <td>327.4401</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:02</th>
      <td>2.439548</td>
      <td>260.9131</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.467305</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.4401</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:03</th>
      <td>2.428338</td>
      <td>260.2850</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.2799</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:04</th>
      <td>2.424815</td>
      <td>259.8925</td>
      <td>244.4245</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.466536</td>
      <td>20.84320</td>
      <td>2.233354</td>
      <td>327.1597</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
df1.shape
```




    (946719, 26)




```python
# Plotting the three sensors' SWaT data in its first stage

def plot (): 
    plt.figure(figsize=(10,6), dpi=350) 
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
```

    None



    
![png](output_16_1.png)
    


#### Exploratory Data Analysis


```python
#If there are missing entries, drop them.
df1.dropna(inplace=True)#,axis=1)  
# Total number of rows and columns 
df1.shape 
```




    (946719, 26)




```python
# Dropping the duplicates 
df1= df1.drop_duplicates()
df1.head(2)   
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>AIT401</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>A</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-22 16:00:00</th>
      <td>2.470294</td>
      <td>261.5804</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.471278</td>
      <td>20.79839</td>
      <td>2.235275</td>
      <td>327.4401</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:01</th>
      <td>2.457163</td>
      <td>261.1879</td>
      <td>244.3284</td>
      <td>8.19008</td>
      <td>306.101</td>
      <td>2.468587</td>
      <td>20.79839</td>
      <td>2.234507</td>
      <td>327.4401</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.68905</td>
      <td>0.001666</td>
      <td>0.001409</td>
      <td>0.001664</td>
      <td>0.0</td>
      <td>10.02948</td>
      <td>0.0</td>
      <td>4.277749</td>
      <td>0.000256</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 26 columns</p>
</div>




```python
# Counting the number of rows after removing duplicates.
df1.shape 

```




    (928898, 26)




```python
# Finding the relations between the variables using the correlation 
plt.figure(figsize=(18,10))
c= df1.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
#c  
```




    <AxesSubplot:>




    
![png](output_21_1.png)
    



```python
# looking the distribution of the data between attack and normal

print(df1.groupby('A')['A'].count()) 
```

    A
    0.0    875250
    1.0     53648
    Name: A, dtype: int64


#### Visualize the dataset 
-  Plotting the number of normal and Attack transactions in the dataset.


```python
#Visualizing the imbalanced dataset
count_classes = pd.value_counts(df1['A'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df1['A'].unique())))#, df1.A.unique()))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations"); 
```


    
![png](output_24_0.png)
    



```python
# count the number of anomalies and normal data points in our dataset 
df1['A'].value_counts()
```




    0.0    875250
    1.0     53648
    Name: A, dtype: int64




```python
# drop the time stamp cols 
#df1= df1.drop('Timestamp', axis=1)

```

#### Normalize using min Max scaler 


```python
# build the scaler model 
# from sklearn.preprocessing import MinMaxScaler 

con_feats = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601'] 
scaler = MinMaxScaler() 
df1[con_feats] = scaler.fit_transform(df1[con_feats])
df1.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIT101</th>
      <th>LIT101</th>
      <th>AIT201</th>
      <th>AIT202</th>
      <th>AIT203</th>
      <th>FIT201</th>
      <th>DPIT301</th>
      <th>FIT301</th>
      <th>LIT301</th>
      <th>AIT401</th>
      <th>...</th>
      <th>AIT504</th>
      <th>FIT501</th>
      <th>FIT502</th>
      <th>FIT503</th>
      <th>FIT504</th>
      <th>PIT501</th>
      <th>PIT502</th>
      <th>PIT503</th>
      <th>FIT601</th>
      <th>A</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-22 16:00:00</th>
      <td>0.894987</td>
      <td>0.160292</td>
      <td>0.730144</td>
      <td>0.732892</td>
      <td>0.073596</td>
      <td>0.874201</td>
      <td>0.462186</td>
      <td>0.940694</td>
      <td>0.182199</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.012283</td>
      <td>0.000948</td>
      <td>0.001035</td>
      <td>0.002179</td>
      <td>0.0</td>
      <td>0.004448</td>
      <td>0.0</td>
      <td>0.005921</td>
      <td>0.000142</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:01</th>
      <td>0.890230</td>
      <td>0.159845</td>
      <td>0.730144</td>
      <td>0.732892</td>
      <td>0.073596</td>
      <td>0.873249</td>
      <td>0.462186</td>
      <td>0.940371</td>
      <td>0.182199</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.012283</td>
      <td>0.000948</td>
      <td>0.001035</td>
      <td>0.002179</td>
      <td>0.0</td>
      <td>0.004448</td>
      <td>0.0</td>
      <td>0.005921</td>
      <td>0.000142</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:02</th>
      <td>0.883848</td>
      <td>0.159533</td>
      <td>0.730144</td>
      <td>0.732892</td>
      <td>0.073596</td>
      <td>0.872796</td>
      <td>0.463182</td>
      <td>0.939886</td>
      <td>0.182199</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.012283</td>
      <td>0.000948</td>
      <td>0.001035</td>
      <td>0.002179</td>
      <td>0.0</td>
      <td>0.004448</td>
      <td>0.0</td>
      <td>0.005921</td>
      <td>0.000142</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:03</th>
      <td>0.879786</td>
      <td>0.158819</td>
      <td>0.730144</td>
      <td>0.732892</td>
      <td>0.073596</td>
      <td>0.872524</td>
      <td>0.463182</td>
      <td>0.939886</td>
      <td>0.182049</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.012283</td>
      <td>0.000948</td>
      <td>0.001035</td>
      <td>0.002179</td>
      <td>0.0</td>
      <td>0.004448</td>
      <td>0.0</td>
      <td>0.005921</td>
      <td>0.000142</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-22 16:00:04</th>
      <td>0.878510</td>
      <td>0.158372</td>
      <td>0.731064</td>
      <td>0.732892</td>
      <td>0.073596</td>
      <td>0.872524</td>
      <td>0.463182</td>
      <td>0.939886</td>
      <td>0.181936</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.012283</td>
      <td>0.000948</td>
      <td>0.001035</td>
      <td>0.002179</td>
      <td>0.0</td>
      <td>0.004448</td>
      <td>0.0</td>
      <td>0.005921</td>
      <td>0.000142</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



#### Split the Data to train and Test


```python
# split the normal data with respect to test and Train 
from sklearn.model_selection import train_test_split 
x_good_train, x_good_test = train_test_split(df1, test_size=0.2, random_state=42)   
```


```python
# min max scale the input data or Standard Scaler  
x_good_train = x_good_train[x_good_train.A == 0] #where normal transactions 
x_good_train = x_good_train.drop(['A'], axis=1) #drop the class columns 

test_y = x_good_test['A'] # save the class column for the test set 
x_good_test = x_good_test.drop(['A'], axis=1) #drop the class column 

#transform to ndarray both train and testing 
x_good_train = x_good_train.values #transform to ndarray 
x_good_test = x_good_test.values 
x_good_train.shape, x_good_test.shape#,x_good_train.shape,test_y.shape   
```




    ((700198, 25), (185780, 25))



### Hyperparamter Tuning 

#### import important libraries 


```python
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
```


```python
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
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 128)               3328      
                                                                     
     dense_1 (Dense)             (None, 32)                4128      
                                                                     
     dense_2 (Dense)             (None, 128)               4224      
                                                                     
     dense_3 (Dense)             (None, 25)                3225      
                                                                     
    =================================================================
    Total params: 14,905
    Trainable params: 14,905
    Non-trainable params: 0
    _________________________________________________________________



```python
x_good_test.shape[1] 
```




    25




```python
import time 
```

- Define the callbacks for checkpoints and early stopping 
- Compile the Autoencoder 
- Train the Autoencoder 


```python

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
```

    Epoch 1/15
    2736/2736 - 13s - loss: 0.0616 - accuracy: 0.7239 - val_loss: 0.0442 - val_accuracy: 0.8318 - 13s/epoch - 5ms/step
    Epoch 2/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9188 - val_loss: 0.0441 - val_accuracy: 0.9523 - 12s/epoch - 5ms/step
    Epoch 3/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9468 - val_loss: 0.0441 - val_accuracy: 0.9507 - 12s/epoch - 5ms/step
    Epoch 4/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9580 - val_loss: 0.0441 - val_accuracy: 0.9640 - 12s/epoch - 5ms/step
    Epoch 5/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9643 - val_loss: 0.0440 - val_accuracy: 0.9660 - 12s/epoch - 5ms/step
    Epoch 6/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9673 - val_loss: 0.0440 - val_accuracy: 0.9680 - 12s/epoch - 5ms/step
    Epoch 7/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9683 - val_loss: 0.0440 - val_accuracy: 0.9664 - 12s/epoch - 5ms/step
    Epoch 8/15
    2736/2736 - 13s - loss: 0.0450 - accuracy: 0.9682 - val_loss: 0.0440 - val_accuracy: 0.9687 - 13s/epoch - 5ms/step
    Epoch 9/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9650 - val_loss: 0.0440 - val_accuracy: 0.9381 - 12s/epoch - 5ms/step
    Epoch 10/15
    2736/2736 - 13s - loss: 0.0450 - accuracy: 0.9697 - val_loss: 0.0440 - val_accuracy: 0.9697 - 13s/epoch - 5ms/step
    Epoch 11/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9692 - val_loss: 0.0440 - val_accuracy: 0.9694 - 12s/epoch - 5ms/step
    Epoch 12/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9703 - val_loss: 0.0440 - val_accuracy: 0.9691 - 12s/epoch - 5ms/step
    Epoch 13/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9687 - val_loss: 0.0440 - val_accuracy: 0.9706 - 12s/epoch - 5ms/step
    Epoch 14/15
    2736/2736 - 13s - loss: 0.0450 - accuracy: 0.9701 - val_loss: 0.0440 - val_accuracy: 0.9650 - 13s/epoch - 5ms/step
    Epoch 15/15
    2736/2736 - 12s - loss: 0.0450 - accuracy: 0.9703 - val_loss: 0.0440 - val_accuracy: 0.9707 - 12s/epoch - 5ms/step
    5806/5806 [==============================] - 9s 1ms/step - loss: 0.0440 - accuracy: 0.9707
    Test loss: 0.04403090104460716
    Accuracy: 0.9707019329071045
    Training time: 196.70048546791077


#### Plot training and test loss


```python
plt.plot(grid.history['loss'])
plt.plot(grid.history['val_loss'])
#plt.plot(grid.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.legend(['loss','val_loss'])#,'accuracy'])
plt.show()  
```


    
![png](output_41_0.png)
    


### Detect Anomalies on test data  
- Data points with higher reconstruction loss are considered anomalies. 
- To calculate the reconstruction loss on test data, predict the test data and calculate the root mean square error between the test data and the reconstructed test data. 

#### 1. Predictions and Computing Reconstruction Error using RMSE  


```python
# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions = model.predict(x_good_test)#,verbose=1)
mse = np.mean(np.power(x_good_test - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,'A': test_y}, index=test_y.index)  
error_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-30 20:16:14</th>
      <td>0.248260</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-02-01 11:34:00</th>
      <td>0.284298</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2015-12-26 02:51:57</th>
      <td>0.302722</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-31 09:15:01</th>
      <td>0.148656</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2015-12-31 11:50:08</th>
      <td>0.205361</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Print memory usage 
model_memory=pd.DataFrame(test_x_predictions)

BYTES_TO_MB_DIV = 0.000001
def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")
    
print_memory_usage_of_data_frame(model_memory) 
```

    Memory usage is 18.578 MB



```python
# find the maximum RMSE values using reconstruction error 
maxValues = error_df.max()
 
print(maxValues) 
```

    Reconstruction_error    0.466866
    A                       1.000000
    dtype: float64


#### Plot Reconsrruction using Error Histogram 
- To determine the exact point using kernel density estimation method 


```python
# To identify the maximum and minimum data point for identifying bins 
import tensorflow as tf 
# reconstruction loss for normal test data
train_loss = tf.keras.losses.mse(test_x_predictions, x_good_test)
trainloss=pd.DataFrame(train_loss)
trainloss.describe()  
 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>185780.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.267254</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.037042</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.021579</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.253889</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.266745</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.286611</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.466866</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```




    Text(0.5, 1.0, 'bins - loss distribution = 108')




    
![png](output_48_1.png)
    


### Model Interpretability
- find the number of anomaly points according to stattical method i.e usinG RMSE loss and bin values 


```python
# change X_tes_scaled to pandas dataframe
data_n = pd.DataFrame(x_good_test, index= test_y.index)#, columns=numerical_cols) 
```


```python
def compute_error_per_dim(point):
    
    initial_pt = np.array(data_n.loc[point,:]).reshape(1,9)
    reconstrcuted_pt = model.predict(initial_pt)
    
    return abs(np.array(initial_pt - reconstrcuted_pt)[0]) 
```


```python
# finding the number of anomalies using reconstruction Error 
outliers = error_df.index[error_df.Reconstruction_error >0.35].tolist()  
number_of_outliers = len(outliers) 
print("Number of elements in the anomalies: ", number_of_outliers)  
```

    Number of elements in the anomalies:  2230


#### Calculate RMSE and MAE stastical method 


```python
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
```

    The performance  of autoencoder:
    
    Mean Absolute Error: 0.2051
    Mean Square Error:0.2673
    Root Mean Square Error: 0.5170
    


#### Classfication report using Stastical method 


```python
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
```

    [[172857   2195]
     [ 10693     35]]
     Accuracy:  0.9306276240714824
     Recall:  0.0032624906785980613
     Precision:  0.01569506726457399
                  precision    recall  f1-score   support
    
             0.0       0.94      0.99      0.96    175052
             1.0       0.02      0.00      0.01     10728
    
        accuracy                           0.93    185780
       macro avg       0.48      0.50      0.48    185780
    weighted avg       0.89      0.93      0.91    185780
    



    <Figure size 288x288 with 0 Axes>


#### Visualizing anomaly points 


```python
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
```




    Text(0.5, 0, 'Time')




    <Figure size 720x432 with 0 Axes>



    
![png](output_58_2.png)
    


#### Mean p-Powered Error for auto encoder 
- To enhance the perfomance of reall and precision  


```python
# to identify the reconstruction error between the Decoder and encoder 
test_x_predictions_mean = model.predict(x_good_test)#,verbose=1)
mse2 = np.mean(np.power(x_good_test - test_x_predictions_mean, 10), axis=1)
error_df_mean = pd.DataFrame({'Reconstruction_error': mse2,'A': test_y}, index=test_y.index)  
error_df_mean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>185780.000000</td>
      <td>185780.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>11.704450</td>
      <td>0.057746</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.012946</td>
      <td>0.233263</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000010</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.088603</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.743879</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.168163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>101.905426</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# find the maximum of each column
maxValues_mean = error_df_mean.max()
 
print(maxValues_mean) 
```

    Reconstruction_error    101.905426
    A                         1.000000
    dtype: float64


#### Plot Reconsrruction Error Histogram 
- to determine the exact point instead of pick up manually 


```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>185780.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>11.704450</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.012946</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.088603</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.743879</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.168163</td>
    </tr>
    <tr>
      <th>max</th>
      <td>101.905426</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```




    Text(0.5, 1.0, 'bins - loss distribution = 29')




    
![png](output_64_1.png)
    



```python
# finding the number of anomalies using reconstruction Error 
outliers_mean = error_df_mean.index[error_df_mean.Reconstruction_error > 20.4].tolist()  
number_of_outliers_mean = len(outliers_mean) 
print("Number of elements in the anomalies: ", number_of_outliers_mean ) 
```

    Number of elements in the anomalies:  13131


#### Classfication report using Mean p-Powered Error


```python
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
```

    [[162146  12906]
     [ 10503    225]]
     Accuracy:  0.8739961244482721
                  precision    recall  f1-score   support
    
             0.0       0.94      0.93      0.93    175052
             1.0       0.02      0.02      0.02     10728
    
        accuracy                           0.87    185780
       macro avg       0.48      0.47      0.48    185780
    weighted avg       0.89      0.87      0.88    185780
    



    <Figure size 288x288 with 0 Axes>


### Dynamic way of identifyig threshold 

#### 1. Logistic regression 

- Define threshold for our model using logistic regression 


```python
# Fit logistic regression model on training set 
from sklearn.linear_model import LogisticRegression 
```


```python
df_new = error_df[['Reconstruction_error', 'A']]
df_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-30 20:16:14</th>
      <td>0.248260</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-02-01 11:34:00</th>
      <td>0.284298</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2015-12-26 02:51:57</th>
      <td>0.302722</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-12-31 09:15:01</th>
      <td>0.148656</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2015-12-31 11:50:08</th>
      <td>0.205361</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



- Reconstruction Error values are shown on the x-axis, and the predicted probability, or anomalies or normal data points, is shown on the y-axis. 

- It is clear that normal probability are related with larger values of reconstruction error (normal data points) 
- The figure also included the threshold values, which were either quite close or 0.15. 


```python
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
```


    
![png](output_73_0.png)
    



```python
# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(df_new.Reconstruction_error, df_new.A, test_size=0.2, 
                                                    random_state=42) 
```


```python
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
```


```python
x_train=x = np.array(x_train)
y_train = np.array(y_train) 
```


```python
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix 
lrt = LogisticRegressionWithThreshold()
lrt.fit(x_train.reshape(-1, 1), y_train)

threshold, optimal_tpr_minus_fpr = lrt.threshold_from_optimal_tpr_minus_fpr(x_train.reshape(-1, 1), y_train)
y_pred = lrt.predict(x_train.reshape(-1, 1), threshold)

threshold, optimal_tpr_minus_fpr  
```




    (0.07251937481898854, 0.63636194937627)




```python
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
```

    [[162146  12906]
     [ 10503    225]]
     Accuracy:  0.9422542792550328
                  precision    recall  f1-score   support
    
             0.0       0.94      1.00      0.97    175052
             1.0       0.00      0.00      0.00     10728
    
        accuracy                           0.94    185780
       macro avg       0.47      0.50      0.49    185780
    weighted avg       0.89      0.94      0.91    185780
    


    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1327: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1327: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /opt/conda/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1327: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



    <Figure size 288x288 with 0 Axes>


### 2. CUSUM  
-  The developed ADS encapsulates a detection algorithm that records the gradual change of the monitored variables, and issues an alert if the deviation is persistent. 
- In the above visual, the anomaly score peaks at time 100 and time 200, which corresponds to points where massive shifts in the time series occur. Setting a minimum threshold for anomaly scores, where anything above a certain threshold corresponds to a change point in the sequence, is the best way to identify individual change points in the series.


```python
# We can take the cumsum of the Reconstruction_error column to see how it changes, and 
# create a new column called Cumsum:
error_df['Cumsum'] = error_df['pred'].cumsum()  
error_df.head()  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
      <th>pred</th>
      <th>Cumsum</th>
      <th>Change</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-30 20:16:14</th>
      <td>0.248260</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2016-02-01 11:34:00</th>
      <td>0.284298</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>+</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2015-12-26 02:51:57</th>
      <td>0.302722</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-31 09:15:01</th>
      <td>0.148656</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>+</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-31 11:50:08</th>
      <td>0.205361</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>+</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Furthermore, we can look into whether the change is positive or negative: 
error_df['Change'] = np.where(error_df['A'] < error_df['pred'].shift(), '+', '-')
error_df.head(5) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
      <th>pred</th>
      <th>Cumsum</th>
      <th>Change</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-30 20:16:14</th>
      <td>0.248260</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2016-02-01 11:34:00</th>
      <td>0.284298</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2015-12-26 02:51:57</th>
      <td>0.302722</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2015-12-31 09:15:01</th>
      <td>0.148656</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2015-12-31 11:50:08</th>
      <td>0.205361</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
  </tbody>
</table>
</div>




```python
# find the maximum of each column using reconstruction error 
maxValues = error_df['Cumsum'].max()
print(maxValues)  
```

    2230





```python
# We can also investigate whether the change is going in the positive or negative direction: 
error_df['Cumsum'] = error_df[['Reconstruction_error','A']].groupby('A').cumsum()
error_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
      <th>pred</th>
      <th>Cumsum</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-30 20:16:14</th>
      <td>0.248260</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.248260</td>
      <td>-</td>
    </tr>
    <tr>
      <th>2016-02-01 11:34:00</th>
      <td>0.284298</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.284298</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-26 02:51:57</th>
      <td>0.302722</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.550983</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-31 09:15:01</th>
      <td>0.148656</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.432954</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-31 11:50:08</th>
      <td>0.205361</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.756344</td>
      <td>+</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-01-01 14:51:43</th>
      <td>0.255589</td>
      <td>0.0</td>
      <td>0</td>
      <td>47569.647043</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-26 12:55:24</th>
      <td>0.261291</td>
      <td>0.0</td>
      <td>0</td>
      <td>47569.908334</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-28 19:49:49</th>
      <td>0.289216</td>
      <td>0.0</td>
      <td>0</td>
      <td>47570.197550</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-27 00:38:54</th>
      <td>0.250955</td>
      <td>0.0</td>
      <td>0</td>
      <td>47570.448505</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2015-12-31 21:38:52</th>
      <td>0.255971</td>
      <td>0.0</td>
      <td>0</td>
      <td>47570.704475</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
<p>185780 rows × 5 columns</p>
</div>




```python
# We can also set an arbitrary threshold and see when the cumsum is above or below it: 

error_df ['Threshold'] = np.where(error_df ['Cumsum'] < 0.4, 'Past Threshold', '+')
error_df.head(2)  

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reconstruction_error</th>
      <th>A</th>
      <th>pred</th>
      <th>Cumsum</th>
      <th>Change</th>
      <th>Threshold</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-30 20:16:14</th>
      <td>0.248260</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
    <tr>
      <th>2016-02-01 11:34:00</th>
      <td>0.284298</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>Past Threshold</td>
    </tr>
  </tbody>
</table>
</div>



#### 


```python
#detect_cusum detects changes in a single column x 
def detect_cusum(x, threshold=1, drift=0, show=True, ax=None):
    print('one')
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
    print('two')

#_is_change loops over all the columns and calls the detect_cusum function for each column
def _is_change(df1):  
    print ('three')
    keep=['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601']

    #preprocessing
    df = df1[keep].astype(float)
    df=(df-df.mean())/df.std() #standardize
    df=df.fillna(-1) #for a given window size, the standardization sometimes yields Nan because the column is constant. Replace those by -1.
    df['A']=0
    #loop over all the columns
    for i in df.columns:
        x = df[i].to_numpy()
        ta=detect_cusum(x,- 0.35, 0.35, True) 
        for j in ta:
            df['A'][j]=1 

    print('kkk')
    #merge the labels back after prediction
    df['A']=df1['A'] 
    df[keep]=df1[keep]#get back unnormalized values 
    print('ok')
    return df  

def main():
    print('five')
    preds=_is_change(df1) 
    print("preds: " ,preds)
    for pred in range(len(preds)):
        #print(preds) 
        point = Point("CUSUM")  
        for col in preds.columns: 
            if col != 'Timestamp':                      
                point.field(col, float(preds.iloc[pred][col]))
                if col== 'label' or col== 'prediction':
                    point.tag(col, float(preds.iloc[pred][col]))
    print('m')
    df=pd.DataFrame(columns=['Timestamp','FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
            'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
            'PIT501','PIT502','PIT503','FIT601','A'])

if __name__ == "__main__":   
    main() 

```

    five



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [1], in <cell line: 67>()
         63     df=pd.DataFrame(columns=['Timestamp','FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','FIT301','LIT301','AIT401',
         64             'AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
         65             'PIT501','PIT502','PIT503','FIT601','A'])
         67 if __name__ == "__main__":   
    ---> 68     main()


    Input In [1], in main()
         50 def main():
         51     print('five')
    ---> 52     preds=_is_change(df1) 
         53     print("preds: " ,preds)
         54     for pred in range(len(preds)):
         55         #print(preds) 


    NameError: name 'df1' is not defined



```python

```
