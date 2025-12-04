#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
data={'name':['hara','vara','thara','suji','asw'],'age':[20,20,12,24,23]}
df=pd.DataFrame(data)
data={'fooststeps':[1000,2000,3000,],inline:True}
df=pd.DataFrame(data)
df

# In[6]:


df

# In[7]:


df.head()

# In[8]:


df.tail(2)


# In[9]:


df[['name','age']]

# In[9]:


import pandas as pd
dat=pd.Series([1,2,4],index=[1,2,3])
print(dat)


# In[6]:


import pandas as pd
 
data = pd.DataFrame(
    {
        "names": ["vamshi", "harshith", "sai", "varshini", "charan", "tarak"],
        "heartbeat": [21, 22, 20, 21, 23, 22],
        "pulse": [100, 110, 120, 130, 140, 150],
    }
)
data = pd.concat([data, pd.Series([1, 2, 3, 4, 5, 6], name="foot steps")], axis=1)
print(data)
 
 

# In[1]:


import pandas as pd
df=pd.DataFrame({'Name':['Ravi','Harshit','Pallavi','varshni','gourangi'],
                 'Steps':[75,85,76,88,79]})
# print(df)
# print("Sum of Steps :",df['Steps'].sum())
# print(min(df["Steps"]))
# print(max(df["Steps"]))
# print("Average :",df['Steps'].mean())
# print("Count :",df['Steps'].count())
result=df['Steps'].agg(['mean','min','max','sum','count'])
print(result)

# In[15]:


df=pd.concat([df,pd.Series([72,80,75,78,74],name='beat')],axis=1)
print(df)
df.loc[0,'beat']==np.nan
df

# In[13]:


df=pd.concat([df,pd.Series([
            "calm",
            "moderate",
            "physical activity",
            "possibly good cardiovascular fitness",
            "moderate"],name='effort')],axis=1)
df




# In[10]:


df.describe()

# In[11]:


df2=pd.DataFrame({'run':[1,2,3,4]})
result=pd.concat(df,df2)
result


# In[12]:


df.loc(0,'beat')=np.nan
df

# In[1]:


import pandas as pd
import numpy as np
df=pd.DataFrame({'Name':['Ravi','Harshit','Pallavi','varshni','gourangi'],
                 'Steps':[75,85,77,88,99]})
print(df)
df.loc[0,'Steps']=np.nan
print("After converting null \n",df)
# n=df.copy()
# df.isnull(
 

# In[3]:


df_index=df.set_index('Name')
print(df_index)
df_index=df_index.reset_index()
print(df_index)

# In[4]:


import numpy as np
data = np.array([
    [72, 3500, 80],
    [75, 4200, 82],
    [70, 3100, 78],
    [68, 2900, 76]
])
heartbeat = data[:, 0]   
steps = data[:, 1]      
pulse = data[:, 2]        
print("Heartbeat:", heartbeat)
print("Steps:", steps)
print("Pulse:", pulse)

# In[5]:


import numpy as np
zer=np.zeros(2,2)
print(zer)
import numpy as np
data = np.full((2, 3), 0)
print(data)

# In[1]:


import numpy as np
health_data = np.array([
    [72, 5000, 1500],
    [78, 6500, 1800],
    [65, 4000, 1200],
    [80, 8000, 2000],
    [75, 5500, 1600]
])
single_element = health_data[2, 1]
print(f"Single element (Steps for the third user): {single_element}")
full_row = health_data[0, :]
print("\nFull row (First user's data):")
print(full_row)
 
# Accessing a full column (e.g., the Heartbeat column)
full_column = health_data[:, 0]
print("\nFull column (Heartbeat):")
print(full_column)
 
# Extracting a range of values from a 1D array (e.g., the first two values of the second row)
selected_row = health_data[1, :]
range_from_row = selected_row[0:2]
print("\nRange of values from the second row (first two values):")
print(range_from_row)
 
# Extracting selected portions of the 2D array (e.g., the Steps and Calories for the first three users)
selected_portion = health_data[:3, 1:3]
print("\nSelected portion (Steps and Calories for the first three users):")
print(selected_portion)
 
# Extracting selected portions using advanced indexing (e.g., Heartbeat of first user and Calories of third user)
advanced_selection = health_data[[0, 2], [0, 2]]
print("\nAdvanced selection (Heartbeat of first user and Calories of third user):")
print(advanced_selection)

# In[2]:


#ctrl+left click in mouse

# In[4]:


#MATHEMATICAL OPERATION
import numpy as np
a=np.array([72,80,20])
b=np.array([76,86,26])
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a**2)
print(np.sqrt(a))
print(np.exp(b))
print(-a)

# In[13]:


#statistical operation
import numpy as np
a=np.array([72,80,70])
print("sum",a.sum())
#mean.median,std,variance,min,max

print(min(a))
print(max(a))
print(np.mean(a))
print(np.median(a))
print(np.std(a))
print(np.var(a))

# In[24]:


import numpy as np
#reshaping
a=np.array([23,3,4,56,67,23,78,90,23,4,567,89])
print(a)

rearray=a.reshape(2,6)
print(rearray)

rearray=rearray.flatten()
print(rearray)

# In[7]:


import numpy as np

heartbeat = np.array([72, 80, 65, 90, 75])
pulse = np.array([70, 78, 64, 88, 74])
steps = np.array([5000, 7000, 6000, 8000, 6500])


zeros_arr = np.zeros(5)
ones_arr = np.ones(5)
range_arr = np.arange(5, 25, 5)

fitness_data = np.column_stack((heartbeat, pulse, steps))
print("2D Fitness Data:\n", fitness_data)

print("Average Heartbeat:", np.mean(heartbeat))
print("Max Steps:", np.max(steps))
print("Std of Pulse:", np.std(pulse))

reshaped = fitness_data.reshape(5, 3) 
flattened = reshaped.flatten()
print("Flattened Array:\n", flattened)

extra_data = np.array([
    [68, 72, 5500],
    [77, 80, 6200],
    [70, 75, 5800],
    [85, 82, 6000],
    [74, 78, 6300]
])

# Vertical stacking
v_stacked = np.vstack((fitness_data, extra_data))
print("Vertical Stack:\n", v_stacked)

# Horizontal stacking (number of rows must match)
h_stacked = np.hstack((fitness_data, extra_data[:, 1:2])) 
print("Horizontal Stack:\n", h_stacked)


# In[4]:


import pandas as pd
import numpy as np
 
data = pd.DataFrame(
    {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "", None, np.nan],
        "Age": [24, 27, 22, 32, 29, 30, 28, 26],
        "Salary": [50000, 60000, 45000, 80000, 70000, 75000, 72000, np.nan],
    }
)
print(data)


print("no of null values in each column:\n", data.isnull().sum())
print(
    "no of empty values in 'Name' column:\n",
    data["Name"].apply(lambda x: x == "").sum(),
)
data = data.dropna()
data = data[data["Name"] != ""]
print(data)
 

# In[8]:


import pandas as pd
 
# Example DataFrame
df = pd.DataFrame({
    'name': ['John', None, 'Sara', 'NaN', '', None],
    'age': [25, 30, 22, 28, 19, 33]
})
 
# Convert string "NaN" and empty strings to actual NaN values
df['name']=df['name'].replace(['NaN', ''], pd.NA)
 
print(df)

# In[14]:


import pandas as pd
import numpy as np
 
 
df = pd.DataFrame({
    'Name': ['Harini', 'abi','raj', None,  '', np.nan],
    'Age': [21, None, None, 24, np.nan,np.nan],
    'Score': [85, 90, 78, 88, 92,100]
})
 
print("DataFrame:")
print(df)
 
# print("\n null values present")
# print(df.isnull().any())
 
# print("\nCount of null values in each column:")
# print(df.isnull().sum())
 
# print("\nCount of empty strings in each column:")
# print((df == "").sum())
# df['Name']=df['Name'].replace('', np.nan)
 
# df_cleaned = df.dropna(subset=['Name'])
 
# print("\nDataFrame after dropping empty and None values in 'Name':")
# print(df_cleaned)
# df['Name']=df['Name'].replace('', np.nan)
 
# df['Name']=df['Name'].fillna('abcde')
 
# print("\nDataFrame after filling missing values:")
# print(df)



df_copy=df.copy
# df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].mean())
# df_copy['Age'] = df_copy['Age'].fillna(20)
df_copy=df_copy.fillna(method='ffill')

# In[12]:


df_copy=df.copy
# df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].mean())
# df_copy['Age'] = df_copy['Age'].fillna(20)
df_copy=df_copy.fillna(method='ffill')

# In[ ]:



