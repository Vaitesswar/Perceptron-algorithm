
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('input1.csv',header = None)
data = data.values

# Data preprocessing
y = data[:,-1]
y = np.reshape(y,[len(y),1])
X = data[:,0:-1]
X = np.hstack((X,np.ones((len(X),1))))
size = np.shape(X)
weights = np.zeros((size[1],1)) # Initialize weights
weights_array = np.array(np.zeros((1,size[1])))

# Training the perceptron
y_pred = (np.matmul(X,weights) > 0)*1 # Finding combined sum
y_pred[y_pred == 0] = -1 # Changing '0' labels to '-1'

while np.all(y == y_pred) != True:
    polarity = (y*y_pred) <= 0 # Find ones that have the wrong polarity
    to_be_added = np.sum(polarity*X*y,0) # Find weights to be added
    to_be_added = np.reshape(to_be_added,[len(to_be_added),1])
    weights += to_be_added
    arr = np.reshape(weights,[1,size[1]])
    weights_array = np.vstack((weights_array,arr))
    y_pred = (np.matmul(X,weights) > 0)*1 # Finding combined sum
    y_pred[y_pred == 0] = -1 # Changing '0' labels to '-1' 
    
weights_array = np.delete(weights_array, [0], axis = 0)
output = np.asarray(weights_array)
np.savetxt("output1.csv", output, delimiter=",",fmt='% 4d')