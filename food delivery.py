#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# set the working directory
os.chdir("E:/documents")
os.getcwd()


# In[3]:


#read the data
train=pd.read_csv("delivery train.csv")


# In[4]:


# checking the data types of the variables
train.dtypes


# # Remove duplicate elements 

# In[4]:


#Removing duplicate elements from the data
print(train.duplicated().sum())
#train.head()
train=train.drop_duplicates()


# # Missing value analysis

# In[5]:


train.isna().sum()


# In[6]:


#finding missing values by observing graph
#pip install missingno
import missingno
missingno.matrix(train)  #there is no missing values as per the graph
("")


# In[10]:


train.head()


# In[13]:


train.describe()


# In[12]:


train.dtypes


# # Changing the varibles to proper datatype

# In[15]:


for i in train.columns:
    print('no of unique value in {} is {} and datatype is {}'.format(i,train[i].nunique(),train[i].dtype))


# we can see that we need to convert the data types of some variables

# In[4]:


lis=['week','center_id','meal_id','emailer_for_promotion','homepage_featured']


# In[5]:


for j in lis:
    train[j]=train[j].astype('object')


# In[21]:


train.dtypes


# # Seperate the variables based on datatypes

# In[6]:


train_num=train.select_dtypes(exclude=['object'])


# In[7]:


train_cat=train.select_dtypes(include=['object'])


# In[10]:


train_num.shape


# In[11]:


train_cat.shape


# # feature generation 

# In[8]:


train['discount']=train['checkout_price']-train['base_price']


# # Dealing with outliers

# In[9]:


low=0.01
high=0.99
train_num.quantile([low,high])


# In[10]:


def outliers(variable):
    low=0.01
    high=0.99
    new=train.quantile([low,high])
    l=new[variable][low]
    print('low-{}'.format(l))
    h=new[variable][high]
    print('high-{}'.format(h))
    lc=len(train[train[variable]<=l])
    hc=len(train[train[variable]>=h])
    t=len(train)
    print('percentage of outliers-{}'.format(((lc+hc)/t)*100))
    #assigning the higher and the lower values in the place of outliers
    train.loc[train[variable]<l,variable]=l
    train.loc[train[variable]>h,variable]=h
    


# In[11]:


sns.boxplot(x=train['num_orders'])
# It has outliers so we have to deal with outliers first


# In[35]:


plt.figure(figsize=(6,4))
sns.distplot(train['num_orders'])


# Since it is a skewed distribution we can say that this variable has outliers and we have to convert into normal distribution form

# In[12]:


outliers("num_orders")


# In[13]:


sns.boxplot(train['num_orders'])
# we remove the outliers to some extent


# we can see that the outliers are removed to some extent

# In[39]:


sns.boxplot(train['discount'])


# In[14]:


outliers('discount')


# In[41]:


sns.boxplot(train['base_price'])


# No outliers present in the base_price variable

# In[128]:


sns.catplot(y='checkout_price',x='emailer_for_promotion',kind='box',data=train)


# # find the most frequently ordered meal

# In[13]:


train.groupby('meal_id')['num_orders'].sum().sort_values(ascending=False)
# This is the short method to find the most ordered meal


# # OR

# In[9]:


#train_ch=train['num_orders']
#train_ch=pd.DataFrame(train_ch)
arr=train['meal_id'].unique()
arr=pd.DataFrame(arr)


# In[10]:


arr.columns=['uniq']
arr.head()


# In[12]:


arr1=[]
for i in arr['uniq']:
    total=0
    for j in range(0,(len(train)-1)):
        if(train['meal_id'].iloc[j]==i):
            total=total+train['num_orders'].iloc[j]
    arr1.append(total)
    


# In[13]:


train['num_orders'].sum()
#total num of orders is 119557485


# In[14]:


arr1=pd.DataFrame(arr1)


# In[15]:


arr2=pd.concat([arr['uniq'],arr1],axis=1)


# In[16]:


arr2=arr2.rename(columns={0:'meal_orders'})


# In[18]:


arr2=arr2.sort_values('meal_orders',ascending= False)


# In[19]:


arr2.head()


# #we can see that order no 2290 is the most repeated item from the all

# In[42]:


#since id column is the unique id for every order we can drop that variable
train=train.drop(columns=['id'])


# # Finding the correlation

# In[45]:


train.head()


# In[44]:


f,ax=plt.subplots(figsize=(7,5))
corr=train.corr()
sns.heatmap(corr)


# In[46]:


corr


# We can observe that the base_price and the checkout_price has the high correlation value of 0.95 so we should drop one of the variable

# In[15]:


train.drop(columns=['checkout_price'],inplace=True)
#checkout price and the base price are highly correlated


# In[48]:


train.head()


# # Univariate analysis of data

# In[50]:


sns.distplot(train['base_price'],kde=True)


# It doesn't follow any distribution 

# In[51]:


sns.distplot(train['discount'],kde=True)


# discount resembles the normal distribution with some outliers

# # Bivariate analysis

# In[53]:


sns.barplot(x='emailer_for_promotion',y='num_orders',data=train)


# #from this we observed that email advertisement has more effect on the num of orders when compared to normal one

# In[54]:


sns.barplot(x='homepage_featured',y='num_orders',data=train)


# homepage_featured items has large effect on num of orders

# In[22]:


sns.catplot(x='emailer_for_promotion',y='num_orders',kind='box',data=train)


# In[72]:


train.columns


# # percentage of orders got discount?

# In[32]:


val=np.where(train['discount']>0)
val=pd.DataFrame(val)
val.shape


# In[26]:


train['discount'].shape


# In[33]:


116101/456548


# By this we have observed that 25 percent of orders get the discount

# In[21]:


train.head()


# In[40]:


train.head()


# # likelihood encoding, impact coding or target coding

# It's a difficult task to encode this center_id and meal_id variables using one_hot_encoding because it increases the dimension of the data. so we can use target encoding to use this variables in the prediction.

# In[55]:


train['center_id'].nunique()


# In[56]:


train['meal_id'].nunique()


# In[65]:


train['week'].nunique()


# In[16]:


# Dealing with categorical variables
# Target encoding of center_id variable
me=train.groupby('center_id')['num_orders'].mean().sort_values(ascending=False)


# In[17]:


# Convert to dictionary
me=me.to_dict()


# In[18]:


# Mapping the mean of the num_orders to the center_id
train['center_id']=train['center_id'].map(me)


# In[60]:


train.head()


# In[19]:


# Target encoding of meal_id variable
me1=train.groupby('meal_id')['num_orders'].mean().sort_values(ascending=False)


# In[20]:


me1=me1.to_dict()


# In[21]:


train['meal_id']=train['meal_id'].map(me1)


# In[64]:


train.head()


# In[22]:


train.drop('week',axis=1,inplace=True)


# # check normality

# In[67]:


plt.hist(train['num_orders'])
("")


# it is a skewed distribution so we need to convert the data into normal distribution by using log transformation

# In[23]:


train['num_orders']=np.log(train['num_orders'])


# In[74]:


plt.hist(train['num_orders'])
("")


# It is converted into normal distribution

# # Autocorrelation in the target variable

# In[83]:


plt.acorr(train['num_orders'],maxlags=10)


# In[24]:


time_1= train['num_orders'].shift(+1).to_frame()
time_2= train['num_orders'].shift(+2).to_frame()
time_3= train['num_orders'].shift(+3).to_frame()


# In[25]:


time=pd.concat([time_1,time_2,time_3,train['num_orders']],axis=1)


# In[26]:


time.columns=['time_1','time_2','time_3','num_orders']


# In[82]:


time.corr()


# In[27]:


train=pd.concat([train,time['time_1']],axis=1)


# In[28]:


train.head()


# In[31]:


train.drop('id',axis=1,inplace=True)


# In[29]:


train.dropna(inplace=True)


# # Scaling the data

# In[36]:


pip install sklearn.preprocessing


# In[32]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
norm=scaler.fit_transform(train)


# In[33]:


norm=pd.DataFrame(norm)


# In[34]:


norm.columns=train.columns


# In[35]:


norm.head()


# In[63]:


x=norm.drop(columns=['num_orders'],axis=1)
y=norm['num_orders']


# In[64]:


x=pd.DataFrame(x_train)
y=pd.DataFrame(y_train)


# In[65]:


# spliting the data into train and test
# since it is a time series data we have to split differently 
train_size=0.7*len(x_train)
train_size=int(train_size)

x_train=x.iloc[0:train_size,:]
x_test=x.iloc[train_size:,:]

y_train=y.iloc[0:train_size,:]
y_test=y.iloc[train_size:,:]


# In[66]:


y_test.head()


# In[53]:


x_train.shape


# In[54]:


y_train.shape


# # Applying the model

# In[55]:


#Random forest modelling
from sklearn.ensemble import RandomForestRegressor
RF_model= RandomForestRegressor(n_estimators=100,max_features=3).fit(x_train,y_train)


# In[67]:


predictions= RF_model.predict(x_test)


# In[68]:


RF_rmse=np.sqrt(mean_squared_error(y_test['num_orders'],predictions))


# In[69]:


RF_rmse


# Random Forest gave the less error compared to other alogorithms
