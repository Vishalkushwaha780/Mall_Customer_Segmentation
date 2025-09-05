#!/usr/bin/env python
# coding: utf-8

# # MALL CUSTOMER SEGMENTATION USING CLUSTERING TECHNIQUES

# ## Problem Statement

# This dataset contains details of 200 customer of a mall with their 5 features which are Customer ID, gender, age, spending score, age and annual income. The sole idea of this project is to identify target customers and develop different marketing strategies to increase sales and customer footfall.

# ### Importing necessary libraries

# In[153]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Importing the dataset using the pandas libraries

# In[155]:


df = pd.read_csv('Mall_Customers.csv')


# ###### Checking the top 5 rows of the dataset

# In[157]:


df.head()


# ### Exploratroy Data Analysis

# In[159]:


df.tail()


# In[160]:


df.sample(5)


# ###### Checking the information about the dataset

# In[162]:


df.info()


# ###### 𝐀𝐭𝐭𝐫𝐢𝐛𝐮𝐭𝐞𝐬 𝐢𝐧𝐟𝐨𝐫𝐦𝐚𝐭𝐢𝐨𝐧:¶
# * CustomerID - Id of the customer of a mall.
# * Gender - Gender of the customer.
# * Age - Age of the customer.
# * Annual Income (k$) - Annual Income of the customer in thousand dollars.
# * Spending Score(1-100) - Spending score of the customer on the scale 1-100.

# ###### Checking the shape of the dataset

# In[165]:


df.shape


# There are 200 rows and 5 columns in the dataset.

# ###### Checking the statistical information

# In[168]:


df.describe()


# ###### Checking the correlation between all the attributes

# In[170]:


df.corr(numeric_only=True)


# Here, we can see that the negative correlation between Age and Spending Score.

# ###### Cheecking for the null values

# In[173]:


df.isnull().sum()


# ###### Checking for the duplicate rows

# In[175]:


df.duplicated().sum()


# #### Univariate Analysis

# ###### Checking the distrubution of the annual income

# In[178]:


plt.figure(figsize=(10,7))
sns.histplot(df['Annual Income (k$)'],color ='b',kde=True)
plt.title("Distribution of Annual Income (k$)",fontsize=20)
plt.xlabel("Annual Income(k$)",fontsize=15)
plt.ylabel("Count",fontsize=15,color='blue')
plt.show()


# ###### distribution of Age 

# In[180]:


plt.figure(figsize=(10,7))
sns.histplot(df['Age'],color ='b',kde=True)
plt.title("Distribution of Age",fontsize=20)
plt.xlabel("Age",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# ###### Distribution of Spending Score

# In[182]:


plt.figure(figsize=(10,7))
sns.histplot(df['Spending Score (1-100)'])
plt.title("Distribution of Spending Score(1-100)",fontsize=20)
plt.xlabel("Spending Score",fontsize=15,color='blue')
plt.ylabel("Count",fontsize=15,color='blue')
plt.show()


# ###### Checking the Percentage of males and females

# In[184]:


plt.figure(figsize=(10,7))
df['Gender'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title("Number of Male and Female Customer",fontsize=20)
plt.show()


# In[245]:


plt.figure(figsize=(10,7))
ax = sns.countplot(x='Gender', data=df)

# Adding count labels on top of bars
for p in ax.containers:
    ax.bar_label(p, fmt='%d', fontsize=12, fontweight='bold')

plt.show()


# #### Bivariate Analysis

# ###### Pair plot for all the attributes

# In[188]:


plt.figure(figsize=(15,15))
sns.pairplot(df)
plt.show()


# #### Multivariate Analysis

# In[190]:


sns.scatterplot(x=df['Annual Income (k$)'],y=df['Spending Score (1-100)'],hue=df['Gender'])
plt.show()


# ###### Data Pre-processing for the models

# Taking all the input attributes in the X variable

# In[193]:


x = df[['Annual Income (k$)','Spending Score (1-100)']]


# In[194]:


x


# ### Building the Models for the Clustering

# #### 1. KMeans Clustering

# ###### importig the KMeans Algorithm

# In[198]:


from sklearn.cluster import KMeans


#  Elbow Method: To get the optimized value of the "K" for how much clusters we need to perform clustering technique.

# In[200]:


wcss=[ ]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(x)
    wcss.append(km.inertia_)


# Checking the WCSS vaues of all the values of the "K"

# In[202]:


wcss


# ###### Making the Elbow curve between the 'K' and 'wcss'

# In[204]:


plt.figure(figsize=(12,12))
plt.plot(range(1,11),wcss,marker='*')
plt.title("WCSS vs K Values",fontsize='20')
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()


# Here, we can see that the elbow is at the k=5. So,we'll go with the K=5.

# ###### Fitting Model Again with K=5

# In[207]:


k_means=KMeans(n_clusters=5)
k_means.fit(x)


# Labels givenn for each clusters

# In[209]:


k_means.labels_


# In[210]:


np.unique(k_means.labels_)


# ###### Predicting the labels for the input data

# In[212]:


y_pred_Km=k_means.predict(x)
y_pred_Km


# Creating the copy of the x in X1 variable

# In[214]:


x1 = x.copy()


# In[215]:


x1.head()


# Adding the y_pred_km as column in X1

# In[217]:


x1['Labels']=y_pred_Km


# ###### Plotting the cluster done by KMeans Clustering

# In[219]:


plt.figure(figsize=(7,8))
sns.scatterplot(x='Annual Income (k$)',y='Spending Score (1-100)',hue='Labels',data=x1,palette=['green','orange','brown','dodgerblue','red'])
plt.show()


# ###### Calculating the Silhouette Score to validate the Clustering Techniques

# In[221]:


from sklearn.metrics import silhouette_score


# In[222]:


Score_km = silhouette_score(x,y_pred_Km)
print(Score_km)


# ###### Now, doing the clustering by adding the Age column as well and also usin the "KMeans ++" for remove the initialization trap

# In[224]:


x2=df[['Annual Income (k$)','Spending Score (1-100)','Age']]


# In[225]:


x2.head()


# In[226]:


wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(x2)
    wcss.append(kmeans.inertia_)


# In[227]:


plt.figure(figsize=(12,12))
plt.plot(range(1,11,1),wcss,linewidth='2',marker='8',color='red')
plt.title("WCSS vs K values")
plt.xlabel("K Values",fontsize=15)
plt.ylabel("WCSS",fontsize=15)
plt.show()


# Again using the K=5

# In[247]:


km2=KMeans(n_clusters=5)
km2.fit(x2)


# In[249]:


y_pred_Km_3d=km2.predict(x2)
print(y_pred_Km_3d)


# In[251]:


x3 = x2
x3['label']=y_pred_Km_3d


# In[253]:


df2=x3


# In[255]:


fig = plt.figure(figsize=(18,18))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2.Age[df2.label == 0], df2["Annual Income (k$)"][df2.label == 0], df2["Spending Score (1-100)"][df2.label == 0], c='purple', s=60)
ax.scatter(df2.Age[df2.label == 1], df2["Annual Income (k$)"][df2.label == 1], df2["Spending Score (1-100)"][df2.label == 1], c='red', s=60)
ax.scatter(df2.Age[df2.label == 2], df2["Annual Income (k$)"][df2.label == 2], df2["Spending Score (1-100)"][df2.label == 2], c='blue', s=60)
ax.scatter(df2.Age[df2.label == 3], df2["Annual Income (k$)"][df2.label == 3], df2["Spending Score (1-100)"][df2.label == 3], c='green', s=60)
ax.scatter(df2.Age[df2.label == 4], df2["Annual Income (k$)"][df2.label == 4], df2["Spending Score (1-100)"][df2.label == 4], c='yellow', s=60)
ax.view_init(35, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# ###### Saving the Kmeans Model for 2d and 3d as well

# In[265]:


import joblib as jb


# Saving the model k_means for 2-Dimensions by Customer_segmentation_2d name 

# In[267]:


jb.dump(k_means,'Customer_segmentation_2d')

