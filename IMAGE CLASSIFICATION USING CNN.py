#!/usr/bin/env python
# coding: utf-8

# ### Developing a Convolutional Neural Network From Scratch For CIFAR10 Datasets Image Classification

# Tensorflow has a wide variety of datasets that we can download and use with just a few lines of code. This is especially helpful when you want to test new models and their implementation and therefore do not want to search for appropriate data for a long time.

# For our exemplae Convolutional Neural Network, we use the CIFAR10 dataset, which is available through Tensorflow. The dataset contains a total of 60,000 images in color, divided into ten different image classes, e.g. horse, duck, or truck. We note that this is a perfect training dataset as each class contains exactly 6,000 images. In classification models, we must always make sure that every class is included in the dataset an equal number of times, if possible. For the test dataset, we take a total of 10,000 images and thus 50,000 images for the training dataset.

# We Imported the required Libraries such as TENSORFLOW,NUMPY,MATPLOTLIB

# In[5]:


import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import numpy as np
import matplotlib.pyplot as plt


# ## LOADING THE DATASETS 

# We are  loading the datasets into 
# ### (x_train,y_train),(x_test,y_test) from tensorflow.keras datasets.cifar10.load_data()

# In[6]:


(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()


# ### Analysing the datasets 

# In[7]:


x_train.shape


# In[8]:


x_test.shape


# In[9]:


x_train[0]


# In[10]:


y_train[0]


# In[11]:


x_test[0]


# In[12]:


y_test[0]


# #### PLOTTING THE FIGURES OF TRAINING DATASETS USING MATPLOTLIB

# In[13]:


plt.figure(figsize = (15,2))
plt.imshow(x_train[0])


# In[14]:


y_train.shape


# In[15]:


y_train[:5]


# #### Reshaping the y_train and y_test into 1D array using Numpy

# In[16]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[17]:


y_test = y_test.reshape(-1,)
y_test[0]


# ### Defining the classes of Images

# In[18]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[19]:


classes[5]


# ### Plotting the train_images and labels using matplotlib for first 10 images using for loop 

# In[20]:


plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # Die CIFAR Labels sind Arrays, deshalb benötigen wir den extra Index
    plt.xlabel(classes[y_train[i]])
plt.show()


# #### Defining the function for plotting images (x[index]) & classes (classes[y[index]]) 

# In[21]:


def plot_sample(x,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])


# In[22]:


plot_sample(x_train,y_train,0)


# In[23]:


plot_sample(x_train,y_train,1)


# ### NORMALISE :
# ### Each of these images is 32×32 pixels in size. The pixels in turn have a value between 0 and 255, where each number represents a color code. Therefore, we divide each pixel value by 255 so that we normalize the pixel values to the range between 0 and 1.

# In[24]:


x_train = x_train / 255
x_test = x_test / 255


# In[25]:


x_train[:5]


# In[26]:


x_test[:5]


# ###  ARTIFICIAL NEURAL NETWORK :
# 
# ### Devoloping ARTIFICIAL NEURAL NETWORK,Compiling, Fitting and Evaluating

# --> The sequential model allows us to specify a neural network, precisely, sequential: from input to output, passing through a series of neural layers, one after the other.
# 
# --> Keras Dense layer is the layer that contains all the neurons that are deeply connected within themselves. 
# 
# --> This means that every neuron in the dense layer takes the input from all the other neurons of the previous layer. 
# 
# --> We can add as many dense layers as required.
# 
# --> model. fit() : fit training data. For supervised learning applications, this accepts two arguments: 
#    
#    the data X and the labels y (e.g. model. fit(X, y) ). 
#    For unsupervised learning applications, this accepts only a single argument, the data X (e.g. model.fit(x_train, y_train, epochs = 5))

# In[27]:


ann = models.Sequential([
    layers.Flatten(input_shape = (32,32,3)),
    layers.Dense(3000, activation ='relu'),
    layers.Dense(1000, activation ='relu'),
    layers.Dense(10, activation ='sigmoid'),
])

ann.compile(optimizer = 'SGD',
           loss ='sparse_categorical_crossentropy',
           metrics = ['accuracy'])

ann.fit(x_train, y_train, epochs = 5)


# In[28]:


ann.evaluate(x_test,y_test)


# In[29]:


# CLASSIFICATION REPORT


# In[30]:


from sklearn.metrics import confusion_matrix, classification_report
y_pred = ann.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test,y_pred_classes))


# ## From the above observations ANN Model has around 50% f1 score and accuracy so to improve f1 score and accuracy of the model we use CNN

# #### In Tensorflow we can now build the Convolutional Neural Network by defining the sequence of each layer. Since we are dealing with relatively small images we will use the stack of Convolutional Layer and Max Pooling Layer twice. The images have, as we already know, 32 height dimensions, 32 width dimensions, and 3 color channels (red, green, blue).
# 
# #### The Convolutional Layer uses first 32 and then 64 filters with a 3×3 kernel as a filter and the Max Pooling Layer searches for the maximum value within a 2×2 matrix.

# In[31]:


cnn = models.Sequential([
    
    #CNN
    layers.Conv2D(filters =32,kernel_size = (3,3), activation = "relu", input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters =64,kernel_size = (3,3), activation = "relu"),
    layers.MaxPooling2D((2,2)),
    
    #DENSE
    layers.Flatten(),
    layers.Dense(64, activation ='relu'),
    layers.Dense(10, activation ='softmax'),
])

cnn.compile(optimizer = 'adam',
           loss ='sparse_categorical_crossentropy',
           metrics = ['accuracy'])

cnn.fit(x_train, y_train, epochs = 10)


# In[32]:


# EVALUATING THE CNN MODEL:

cnn.evaluate(x_test,y_test)


# In[33]:


# PLOTTING THE TEST_IMAGES USING PLOT_SAMPLE FUNCTION WHICH WE CREATED EARLIER. 

plot_sample(x_test,y_test,0)


# In[34]:


plot_sample(x_test,y_test,1)


# ## Predicting the x_test values to get y_pred values 

# In[35]:


y_pred = cnn.predict(x_test)
y_pred[:5]


# ##### Argmax is an operation that finds the argument that gives the maximum value from a target function. Argmax is most commonly used in machine learning for finding the class with the largest predicted probability. Argmax can be implemented manually, although the argmax() NumPy function is preferred in practice.
# 
# ##### Creating y_classes using argmax() for each element in y_pred

# In[48]:


# PREDICTED VALUES 

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[49]:


# TEST_LABEL VALUES 

y_test[:5]


# #### Checking whether the test labels and and predicted lables are same or not

# In[50]:


# ACTUAL IMAGE AND LABEL 

plot_sample(x_test,y_test,3)


# In[51]:


# PREDICTED VALUES 

classes[y_classes[3]]


# In[52]:


# ACTUAL IMAGE AND LABEL

plot_sample(x_test,y_test,100)


# In[53]:


# PREDICTED VALUES 

(classes[y_classes[100]])


# In[55]:


# ACTUAL IMAGE AND LABEL

plot_sample(x_test,y_test,5000)


# In[57]:


# PREDICTED VALUES 

(classes[y_classes[5000]])


# ### From the above observations we can see the actual test image and labels and the Predicted labels are almost same Hence our CNN Model is not Predicting a bad value

# In[58]:


# CLASSIFICATION REPORT

print("Classification Reporrt: \n", classification_report(y_test,y_classes))


# ## CONCLUSION: 
# 
# ### Our prediction of the image class is correct in about 70% of the cases. This is not a bad value, but not a particularly good one either. If we want to increase this even further, we could have the Convolutional Neural Network trained for more epochs or possibly configure the dense layers even differently.

# In[ ]:




