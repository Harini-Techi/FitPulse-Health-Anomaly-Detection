#!/usr/bin/env python
# coding: utf-8

# In[4]:


import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")
fig.show()

# In[2]:


!pip install plotly



# In[5]:


import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")
fig.show()

# In[ ]:


#import plotly.express as px
importing plotly express library and giving shortcut name 
 
this is a module in the plotly library used for quick and simple charts which helps in realtime usages
 
