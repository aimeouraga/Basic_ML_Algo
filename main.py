#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from Linear_Regression import linear_regression
import Logistic_Regression as LoRe
import sklearn
from sklearn.datasets import make_classification
import Plots


xtrain = np.linspace(0,1,50)
ytrain = xtrain + np.random.normal(0, 0.1, (50,))
xtrain = xtrain.reshape(-1,1)
ytrain = ytrain.reshape(-1,1)


Plots.visual_data_lin(xtrain, ytrain)


X, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)


X_train, y_train, X_test, y_test = LoRe.train_test_split(X, y)


model1 = linear_regression()
model2 = LoRe.LogisticRegression(lr=0.01, n_epochs=10000)

def main():
    
    model1.fit_gradient_descent(xtrain, ytrain, nb_epochs = 30, lr=0.1)
    model1.fit_stochastic_gradient_descent(xtrain, ytrain, nb_epochs = 30, lr=0.01)
    model1.fit_mini_batch_gradient_descent(xtrain, ytrain, batch_size = 3, nb_epochs = 30, lr=0.1)
    model2.fit(X_train, y_train)
    
if __name__ == "__main__":
    main()


