#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np

def plot(X, y, theta, epoch, plot_every=2):
  """Plotting function for features and targets"""
  if plot_every is not None and epoch % plot_every == 0:
    xtest = np.linspace(0, 1, X.shape[0]).reshape(-1,1)
    ypred = (xtest @ theta).reshape(-1,1)
    plt.scatter(X, y, marker="+")
    plt.xlabel("feature")
    plt.ylabel("target")
    plt.plot(xtest, ypred, color="orange")
    plt.show()
    
def vizualize_data(X, y, col1, col2):
    plt.scatter(X[:,col1], X[:,col2], c=y)
    plt.xlabel('features 1')
    plt.ylabel('feature 2')
    plt.show()
    
def visual_data_lin(x, y):
    plt.scatter(x, y, marker='+')
    plt.title("Pot of data for Linear regression")
    plt.show()


