#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import os

data = pd.read_csv(r"./dataset/Dataset.csv")

data = data.replace('?', np.nan)



# In[17]:


data


# In[18]:


print('********before missing value*********')
print(data.isna().sum())


# In[23]:


#data['cp']=data['cp'].fillna(data['cp'].mode()[0])
data['temperature']=data['temperature'].fillna(data['temperature'].mean())
data['humidity']=data['humidity'].fillna(data['humidity'].mean())
data['ph']=data['ph'].fillna(data['ph'].mean())
data['rainfall']=data['rainfall'].fillna(data['rainfall'].mean())


# In[25]:


print('*****after clear missing value*****')
print(data.isna().sum())


# In[26]:


data.to_csv(os.path.join('./preprocess_dataset/Preprocessed_Dataset.csv'), index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




