# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:40:46 2018

@author: niloufar.valinejad
"""

# Import needed libraries
import pandas as pd
import numpy as np

# Read files:
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

# Add one extra attribute to data and label them as train ot test
train['source']='train'
test['source']='test'

#Combine test and train data for implementing feature engineering on them all at once
print("----------------------------------------------------------\n")
data = pd.concat([train, test],ignore_index=True)
print("Shape of Train data%s, Test data%s, Combined data%s"%(train.shape, test.shape, data.shape))
print("----------------------------------------------------------\n")

# Display Null cells (find missing values)
''' Item_Outlet_Sales is the target variable (Label) which only is available in train data,
It doesnt consider as missing values '''
display_null = data.apply(lambda x: sum(x.isnull()))
print("Number of Null cells in each attribute:\n%s"%display_null)

print("-----------------------------------------\nBasic statistics for numerical variables %s"%data.describe())

# Nominal (categorical) variable. 
# number of unique values in each of them. (Variety)
categorical_columns = data.apply(lambda x: len(x.unique()))
print("-----------------------------------------\nNumber of unique values in each Categorical Variable %s"%categorical_columns)
print("\n--------------------------------------------------------------------")

#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())
    
#Determine the average weight per item:
item_avg_weight = pd.pivot_table(data, index='Item_Identifier', values='Item_Weight')
print(item_avg_weight)

#Get a boolean variable specifying missing Item_Weight values
#Impute data and check missing-values before and after imputation to confirm
'''print('Orignal #missing: %d'% sum(data['Item_Weight'].isnull()))
print(data['Item_Weight'])

#This line has problem I couldnt solve
data['Item_Weight'] = data['Item_Weight'].fillna(item_avg_weight)

print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))
print(data['Item_Weight'])'''


#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull()
print(miss_bool[miss_bool==True]) 

#Impute data and check #missing values before and after imputation to confirm
print("Orignal #missing: %d"% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
print("Final #missing: %d"% sum(data['Item_Weight'].isnull()))
print(data.loc[miss_bool,'Item_Weight'])











