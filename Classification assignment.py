#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


df=pd.read_csv("C:\\Users\\asus\\Downloads\\zoo.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[11]:


duplicates = df.animal_name.value_counts()
duplicates[duplicates > 1]


# In[12]:


frog = df.loc[df['animal_name'] == 'frog']
frog


# In[18]:


color_list = [("red" if i == 1 else "blue" if i == 0 else "yellow" ) for i in df.hair]
unique_color = list(set(color_list))
unique_color


# In[19]:


pd.plotting.scatter_matrix(df.iloc[:,:7],
                                       c=color_list,
                                       figsize= [20,20],
                                       diagonal='hist',
                                       alpha=1,
                                       s = 300,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()


# In[20]:


sns.countplot(x="hair", data=df)
plt.xlabel("Hair")
plt.ylabel("Count")
plt.show()
df.loc[:,'hair'].value_counts()


# In[27]:


df1=pd.read_csv("C:\\Users\\asus\\Downloads\\class.csv")


# In[35]:


D = pd.merge(df,df1,how='left',left_on='class_type',right_on='Class_Number')


# In[36]:


D.head()


# In[37]:


type_list = [i for i in D.class_type]
unique_type = list(set(type_list))
unique_type


# In[38]:


sns.factorplot('Class_Type', data=D, kind="count",size = 5,aspect = 2)


# In[23]:


from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:





# In[ ]:




