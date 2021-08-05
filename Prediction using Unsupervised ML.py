#!/usr/bin/env python
# coding: utf-8

# # NAME -  SONALI PATIL
# 
# # TASK 2 - Prediction using Unsupervised ML (Level - Beginner)
# 

# In[1]:


## IMPORTING ALL REQUIRED LABRIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[19]:


# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() 


# # Finding the optimum number of clusters for K Means? How does one  determine the value of K?
# 

# In[5]:


#  Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[6]:


# You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs.
#This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.


# In[7]:


# Applying kmeans to the dataset / Creating the kmeans classifier

# choose the number of clusters as ** '3**'.


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# # Visualization

# In[13]:


# Visualising the clusters - On the first two columns

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend(loc='upper right')
plt.grid()


# # THANK YOU !!!!!!!!!

# In[ ]:




