#!/usr/bin/env python
# coding: utf-8

# # HR Data Cleaning for Analysis
# 

# In[1]:


# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[2]:


# importing the data set
HR_dataset=pd.read_csv("HR-Employee-Attrition.csv")


# In[3]:


HR_dataset.head() # checking the dataset



# In[4]:


HR_dataset.tail()#checking the tail just to check , Is there any sequence from top to bottom


# In[5]:


HR_dataset.sample(5)


# In[6]:


HR_dataset.shape


# In[7]:


HR_dataset.size


# In[8]:


HR_dataset.info() #to see datatype of different column.a


# In[9]:


HR_dataset.describe() # to check statistics of numeric column


# In[10]:


HR_dataset.isnull().sum()


# In[11]:


#so in the data set there is no missing value and has more features than required,
Columns=HR_dataset.columns


# In[12]:


# Creating missing values manually into 5+ selected columns
HR_dataset.loc[5:10, 'Age'] = np.nan
HR_dataset.loc[15:20, 'Department'] = np.nan
HR_dataset.loc[[3, 8, 14], 'MonthlyIncome'] = np.nan
HR_dataset.loc[HR_dataset.sample(frac=0.05).index, 'JobSatisfaction'] = np.nan  # this code will randomly select 5% of data and make it null
HR_dataset.loc[HR_dataset.sample(frac=0.05).index, 'Education'] = np.nan


# In[13]:


HR_dataset.isnull().sum()


# In[14]:


HR_dataset['MonthlyIncome'].dtype


# In[15]:


'''so now we have dataset which has missing values.I will impute age by mean , department by mode ,
monthly income by median,job satisfaction and education field by mode'''
get_ipython().system('pip install scikit-learn')



# In[16]:


HR_dataset["MonthlyIncome"]


# In[17]:


print(type(HR_dataset))


# In[18]:


imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(HR_dataset[['Age']])

HR_dataset['Age']=imputer.transform(HR_dataset[['Age']])
print(type(HR_dataset))
HR_dataset_imputed=pd.DataFrame(HR_dataset,columns=HR_dataset.columns,index=HR_dataset.index)


# In[19]:


#after using fit_transform the dataframe is converted to narray
print(type(HR_dataset))
HR_dataset.columns


# In[20]:


HR_dataset.isnull().sum()


# In[21]:


cols_to_impute = ['Department', 'JobSatisfaction', 'Education']
imputer2=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer2.fit(HR_dataset[cols_to_impute])
HR_dataset[cols_to_impute]=imputer2.transform(HR_dataset[cols_to_impute])


# In[22]:


HR_dataset.isnull().sum()


# In[23]:


imputer3=SimpleImputer(missing_values=np.nan,strategy='median')
imputer3.fit(HR_dataset[["MonthlyIncome"]])
HR_dataset["MonthlyIncome"]=imputer3.transform(HR_dataset[["MonthlyIncome"]])


# In[24]:


HR_dataset.isnull().sum()


# Correcting the datatypes 
# 
# 

# In[25]:


HR_dataset.info()


# In[26]:


#converting education from integer to string
education_map = {
    1: 'Below College',
    2: 'College',
    3: 'Bachelor',
    4: 'Master',
    5: 'Doctorate'
}

HR_dataset['Education'] = HR_dataset['Education'].map(education_map)


# In[27]:


#converting WorkLifeBalance from [1,2,3,4] to [Bad, Good, Better,Best]
balance_labels = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
HR_dataset['WorkLifeBalance'] = HR_dataset['WorkLifeBalance'].map(balance_labels)
HR_dataset['WorkLifeBalance'].value_counts


# In[28]:


HR_dataset.info()


# In[29]:


HR_dataset['RelationshipSatisfaction']


# In[30]:


#Converting Relationship status into object type
relationship_map = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
}

HR_dataset['RelationshipSatisfaction'] = HR_dataset['RelationshipSatisfaction'].map(relationship_map)
HR_dataset.info()


# In[31]:


# lets convert Monthly income from int to float
HR_dataset['MonthlyIncome'] = HR_dataset['MonthlyIncome'].astype(float)
print(HR_dataset['MonthlyIncome'])


# Encoding categorical columns
# Categorical Columns mention in question set as input Department, Gender,Education

# In[32]:


HR_dataset["Education"]


# In[33]:


HR_dataset["Department"].values


# In[34]:


HR_dataset.columns.get_loc("Department")


# In[35]:


#so here every column has nominal category so I have to one hot encoding
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),["Department"])],remainder='passthrough')

HR_dataset=ct.fit_transform(HR_dataset)
HR_dataset






# In[36]:





# In[ ]:




