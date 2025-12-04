#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#clean steps column
data['heart_beat_per_minute']=pd.to_numeric(data['heart_beat_per_minute'],errors='coerce')
print(data.head())
data['heart_beat_per_minute']=data['heart_beat_per_minute'].fillna(0).astype(int)
print(data.head())
