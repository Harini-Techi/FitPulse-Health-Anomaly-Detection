#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install openpyxl


# In[1]:


import pandas as pd
import numpy as np
data={
    'name':['a','b','c','d','e','f'],
    'age':[12,13,15,17,45,68],
    'Heart_Rate': [78,82,'',76,60,90],
    'Steps': [9200,10500,3000,None,500,10000],
    'Calories': [240,np.nan,220,235,80,275]
}
df=pd.DataFrame(data)
print(df)
df.to_excel("fitpulse_health_data.xlsx",index=False)

# In[3]:


import os
print(os.getcwd())


# In[4]:


import os
print(os.listdir())


# In[4]:


import pandas as pd
df=pd.read_excel(r"C:\Users\KN\fitpulse_health_data.xlsx")
print(df.head())
df.isnull().sum()

# In[5]:


df.isnull().sum()

# In[13]:


heart=df['heart_beat_per_minute'].to_numpy()
print("haert shape",heart.shape)


# In[8]:


import pandas as pd
df=pd.read_excel(r"C:\Users\KN\fitpulse_health_data.xlsx")
df.head()

# In[31]:


import pandas as pd

# convert gender in int

df=pd.read_excel(r"C:\Users\KN\fitpulse_health_data.xlsx")
print(df)

# df.isna().sum()

# clean the data
df['pulse_rate']=pd.to_numeric(df['pulse_rate'],errors='coerce')
print(df)


# fill missing and convert to int

df['pulse_rate']=df['pulse_rate'].fillna(0).astype(int)
print(df)
df['Gender'] = df['Gender'].astype(str)


# # # replace gender in int

df['Gender'] = (
    df['Gender']
    .replace({'M': 1, 'F': 0})     # replace letters
    .apply(pd.to_numeric, errors='coerce')  # convert safely
    .fillna(1)                     # handle NaN
    .astype(int)                   # finally make sure it's int
)
print(df)
# df.loc[4,'Gender']=0
df.to_excel(r"C:\Users\KN\fitpulse_health_data.xlsx",index=False)
print(df)




gender_map={'male':1,'female':0,'m':1,'f':0}
df['gender_code']=(
    df['Gender']
        .astype(str)
        .str.strip()   
        .str.lower()
        .map(gender_map).fillna(2).astype(int))
print(df[['Gender', 'gender_code']])



# upto 60-Low

# 60-100->Normal

# 100----->High
 
low=df[(df['heart_beat_per_minute']<60)]
print(low)


def heartscore(df):
    df['heartscore']=df['heart_beat_per_minute'].apply(
        lambda x:'low' if x<60 else('medium' if x==60 else 'high'))
    print(df)
heartscore(df)




# Task :A simple scatter plot to check correlation between steps and heart rate
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(df['steps'],df['heart_beat_per_minute'],color='b')
plt.grid(True)
plt.show()

# In[27]:


df.replace(np.nan,'unknownk')

# In[33]:


import matplotlib.pyplot as plt
plt.scatter(df.age,df.Calories,color='b')
plt.xlabel('AGE')
plt.ylabel('CALORIES')
plt.show()

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df=pd.read_excel(r"C:\Users\KN\fitpulse_health_data.xlsx") 
 
# do forward fill to remove NaNs or nulls
df.ffill(inplace=True)
 
# # make a bar chart comparing age groups and their heart_beat_per_minute
plt.figure(figsize=(4, 3))
plt.bar(df['age_group'].to_numpy(), df['heart_beat_per_minute'].to_numpy(), color='r', label='Age to BPM Comparison')
plt.title('Age to BPM Comparison')
plt.xlabel('Age Group')
plt.ylabel('BPM')
plt.grid(True)
plt.legend()
plt.show()
 
# make a bar chart comparing age groups and their pulse_rate
plt.figure(figsize=(4, 3))
plt.bar(df['age_group'].to_numpy(), df['pulse_rate'].to_numpy(), color='g', label='Age to Pulse Rate Comparison')
plt.title('Age to Pulse Rate Comparison')
plt.xlabel('Age Group')
plt.ylabel('Pulse Rate')
plt.grid(True)
plt.legend()
plt.show()
 
# make a bar chart comparing age groups and their steps
plt.figure(figsize=(4, 3))
plt.bar(df['age_group'].to_numpy(), df['steps'].to_numpy(), color='b', label='Age to Steps Comparison')
plt.title('Age to Steps Comparison')
plt.xlabel('Age Group')
plt.ylabel('Steps')
plt.grid(True)
plt.legend()
plt.show()
 
# make a pie chart showing age group frequency
plt.figure(figsize=(4, 3))
plt.pie(df['age_group'].value_counts(), labels=df['age_group'].value_counts().index, autopct='%1.1f%%', shadow=True)
plt.title('Age Groups and Their Frequency')
plt.show()
 

 

# In[3]:


#Correlation between heart rate and steps 
# heart rate and steps 
# 1.>0.5=>String positive correlation
# 2.< -0.5=>String negative correlation
# 3. near 0 -> weak or no correlation
import pandas as pd
df=pd.read_excel(r"C:\Users\KN\fitpulse_health_data.xlsx")
correlation=df['steps'].corr(df['heart_beat_per_minute'])
print(f"Correlation between Steps and Heart Rate :{correlation:.3f}")

# In[5]:


avg_heart_gender = df.groupby('Gender')['heart_beat_per_minute'].mean()
 
print("Average Heart Rate by Gender:")
print(avg_heart_gender)

# In[ ]:



