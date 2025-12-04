#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#sample dataset
data=pd.DataFrame({"area":[1000,1500,2500,6000],
                   "pp":[10,15,20,50]})

#feature->x and target->y
x=data[['area']]
y=data[['pp']]

#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)

#predict
pred=model.predict(x_test)
print("pridicted price",pred)
print("MSE",mean_squared_error(y_test,pred))

# In[7]:




# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
 
# Create age range
ages = np.arange(1, 81) # Age 1 to 80
 
heights = []
 
for age in ages:
    if age <= 30:
        # Height increases linearly with some noise
        height = 50 + age * 2 + np.random.normal(0, 1)
    elif 30 < age < 55:
        # Height stays stable
        height = 110 + np.random.normal(0, 1)
    else:
        # Height decreases after 55
        height = 110 - (age - 55) * 1.5 + np.random.normal(0, 1)
 
    heights.append(height)
 
# Create DataFrame
df = pd.DataFrame({
    'age': ages,
    'height': heights
})
 
print(df.head())
print(df.tail())
 
#features->x and Target->y
x=df[['age']]
y=df[['height']]
 
#spilt data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
 
#predict
pred=model.predict(x_test)
 
print("Predicted height ",pred)
 
print("MSE",mean_squared_error(y_test,pred))



#plot

plt.figure(figsize=(6, 4))
plt.scatter(x_test, y_test, color='red', label='True', alpha=0.5)
plt.plot(x_test, pred, color='blue', label='Pred', alpha=0.5)
plt.grid(True)
plt.legend()
plt.title('Predicted vs True')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

# In[6]:


import pandas as pd
df=pd.read_excel(r"presentatio0n.xlsx")
print(df.head())
#cleaning gae column
df['age']=pd.to_numeric(df['age'],errors='coerce')
df['age']=df['age'].fillna(df['age'].mean()).astype(int)
df
df['gender']=df['gender'].replace({'m':1,'f':0})
df
df['gender']=pd.to_numeric(df['gender'],errors='coerce')
df
df.loc[4,'gender']=0
df['gender']=df['gender'].ffill().astype(int)
df
df['heart_beat']=df['heart_beat'].fillna(df['heart_beat'].mean())


def heartscore(df):
    df['heartscore']=df['heart_beat'].apply(
        lambda x:'low' if x<90 else('med' if x==90 else 'high'))
    #return df['heartscore']
heartscore(df)
df

c=df['heart_beat'].corr(df['gender'])
print(f"bb:{c:.3f}")

m=df.groupby('gender')['heart_beat'].mean()
print(m)
df


import pandas as pd
heart_data = pd.read_csv('/content/heart_disease_data.csv')
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
sex_data = heart_data['sex'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(sex_data)
heart_data['sex_cluster'] = cluster_labels
 
print("Original sex values:", heart_data['sex'].value_counts().sort_index())
print("Cluster distribution:", heart_data['sex_cluster'].value_counts().sort_index())
print("Cluster centers:", kmeans.cluster_centers_)
#plot 1 original distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
heart_data['sex'].value_counts().sort_index().plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Original Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
# Plot 2: Cluster distribution
plt.subplot(1, 2, 2)
heart_data['sex_cluster'].value_counts().sort_index().plot(kind='bar', color=['lightgreen', 'orange'])
plt.title('Cluster Distribution')
plt.xlabel('Cluster Label')
plt.ylabel('Count')
plt.show()

# In[ ]:




# In[ ]:




# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import pandas as pd

# -----------------------------
# 1. Create DataFrame
# -----------------------------
data = {
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "HeartRate": [85, 72, 95, 65, 88, 70, 92, 68, 99, 63],
    "PulseRate": [90, 80, 100, 70, 92, 78, 97, 75, 105, 72],
    "Steps": [4000, 3500, 6000, 3000, 5000, 3200, 5800, 3100, 6500, 2900]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Save to Excel
# -----------------------------
df.to_excel("gender_data.xlsx", index=False)

print("gender_data.xlsx created successfully!")


# -----------------------------
# 1. READ YOUR DATA
# -----------------------------
df = pd.read_excel("gender_data.xlsx")   # your file

# -----------------------------
# 2. ENCODE GENDER (Male=1, Female=0)
# -----------------------------
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])

# -----------------------------
# 3. SELECT FEATURES (X) AND TARGET (y)
# -----------------------------
X = df[['HeartRate', 'PulseRate', 'Steps']]
y = df['Gender_encoded']

# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------
# 5. MODEL TRAINING
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. PREDICT
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. ACCURACY
# -----------------------------
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# -----------------------------
# 8. SHOW COEFF & INTERCEPT
# -----------------------------
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

df

# In[ ]:



