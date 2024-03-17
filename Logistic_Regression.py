#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Import packages
import numpy as np
import matplotlib.pyplot as plt


def train_test_split(X,y):
    '''
    this function takes as input the sample X and the corresponding features y
    and output the training and test set
    '''
    np.random.seed(10)

    train_size = 0.8
    n = int(len(X)*train_size)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_idx = indices[: n]
    test_idx = indices[n:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test

class LogisticRegression:
    
    def __init__(self,lr,n_epochs):
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = []
        self.w = None
        self.weight = []
    
    #This following function to add a column vector of 1 in our matrix X 
    def add_ones(self, x):
        return np.hstack( (np.ones((x.shape[0],1)), x))
    
    #This function is to compute the sigmoid function
    def sigmoid(self, x):
        if x.shape[1] < self.w.shape[0]:
            x = self.add_ones(x)
        z = x @ self.w
        return np.divide( 1, 1 + np.exp(-z) )
    
    #Here is the cross entropy function 
    def cross_entropy(self, x, y_true):
        y_pred = self.sigmoid(x)
        loss = (-1/y_true.shape[0]) * np.sum( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )
        return loss
    
    #This function is to compute a predict value but putting them between 0 and 1
    def predict_proba(self,x):
        proba = self.sigmoid(x)
        return proba
    
    #This function is to compute the output, our classes 0 or 1
    def predict(self,x):
        probas = self.predict_proba(x)
        output = (probas >= 0.5).astype(int)
        return output
    
    #Here we use gradient descent algorithm to implement the classification 
    def fit(self,x,y):
        x = self.add_ones(x)
        if np.ndim(y) == 1:
            y = y.reshape((-1,1))

        self.w = np.zeros((x.shape[1],1))

        for epoch in range(self.n_epochs):
            y_pred = self.predict_proba(x)

            grad = (-1/y_pred.shape[0]) * x.T @ (y - y_pred)

            self.w = self.w - self.lr * grad

            loss = self.cross_entropy(x, y)
            self.train_losses.append(loss)

            if epoch%1000 == 0:
                print(f'loss for epoch {epoch}  : {loss}')

        plt.plot(self.train_losses)
        plt.title("Loss for Logistic Regression") 
        plt.legend()
        plt.show()       
    #Here is the function to compute the accuracy
    def accuracy(self,y_true, y_pred):
        y_true = y_true.reshape((-1,1))
        acc = (np.sum((y_true == y_pred).astype(int))) / (y_true.shape[0])
        
