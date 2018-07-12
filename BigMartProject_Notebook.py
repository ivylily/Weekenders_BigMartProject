
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#Import mode function:
from scipy.stats import mode


# In[2]:


# Read files using pandas library:
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")


# In[3]:


#notice this file data types. They are DataFrames 
print(type(test))


# In[4]:


# Add one extra attribute to data and label them as train ot test
train['source']='train'
test['source']='test'


# In[5]:


#Combine test and train data to implement feature engineering on them at once (they will be separated at the end)
print("----------------------------------------------------------\n")
data = pd.concat([train, test],ignore_index=True, sort=True)
print("Shape of Train data%s, Test data%s, Combined data%s"%(train.shape, test.shape, data.shape))
print("----------------------------------------------------------\n")


# In[6]:


# Display Null cells (find missing values)
''' Item_Outlet_Sales is the target variable (Label) which is only available in train data,
It isn't consider as missing values as the quantity null is the quantity supposed to be in the test data. '''
display_null = data.apply(lambda x: sum(x.isnull()))
print("Number of Null cells in each attribute:\n%s"%display_null)

print("-----------------------------------------\nBasic statistics for numerical variables\n%s"%data.describe())


# In[7]:


# Nominal (categorical) variable. 
# number of unique values in each of them. (Variety)
categorical_columns = data.apply(lambda x: len(x.unique()))
print("-----------------------------------------\nNumber of unique values in each Categorical Variable %s"%categorical_columns)
print("\n--------------------------------------------------------------------")


# In[8]:


#Filter categorical variables
#Note they are filtered by dtype object because strings it use strings pointers (references) instead of values.
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())


# #Findings from last point:
# #Item_Fat_Content was mispelled as LF and low fat instead of Low Fat

# DATA CLEANING - 

# In[10]:


#Determine the average weight per item:
item_avg_weight = pd.pivot_table(data, index='Item_Identifier', values='Item_Weight')
print(item_avg_weight)


# In[11]:


#Get a boolean variable specifying missing Item_Weight values
#Impute data and check missing-values before and after imputation to confirm
'''print('Orignal #missing: %d'% sum(data['Item_Weight'].isnull()))
print(data['Item_Weight'])

#This line has problem I couldnt solve
data['Item_Weight'] = data['Item_Weight'].fillna(item_avg_weight)

print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))
print(data['Item_Weight'])'''


# In[12]:


#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull()
print(miss_bool[miss_bool==True])


# In[13]:


#Impute data and check #missing values before and after imputation to confirm
print("Orignal #missing: %d"% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
print("Final #missing: %d"% sum(data['Item_Weight'].isnull()))
print(data.loc[miss_bool,'Item_Weight'])


# In[30]:


#Imputing Outlet_Size (replacing missing data with substituted values)
#Determing the mode for each OutLet_Type
#Do not heed warning
#Original code had and error (something changed with new version) solution found in this thread https://goo.gl/xDp6u3
#Remember this for presentation
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]), fill_value=0)
print('Mode for each Outlet_Type')
print(outlet_size_mode)


# In[22]:


#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Size'].isnull()


# In[31]:


#Impute data and check # missing values before and after imputation to confirm
print('\nOriginal # missing: %d'%sum(miss_bool))
data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

print(sum(data['Outlet_Size'].isnull()))


# In[ ]:


#Create some variables using the existing ones


# In[32]:


data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')


# In[33]:


#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')


# In[34]:


#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)
print('Number of 0 values initially: %d'%sum(miss_bool))


# In[36]:


#Notice: visibility_avg is a dataframe and will not work per se here
#Change in code based on https://goo.gl/UXhJTx
#Remember this for presentation
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))


# In[38]:


#Determine another variable with means ratio saving in new column
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
#Here we can see the data with it's new column
print(data['Item_Visibility_MeanRatio'].describe())

