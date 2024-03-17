#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt


class linear_regression:
    
    def __init__(self):
        pass
        
    def linear_function(self, x, beta):
        return x @ beta
    
    def mean_square_error(self, y, y_pred):
        return (1/y.shape[0]) * np.sum( (y - y_pred)**2 )
    
    def gradient(self, x, y, beta):
        return (-2/y.shape[0]) * x.T @ ( y - self.linear_function(x,beta) )
    
    def initialize(self, x):
        D = x.shape[1]
        return np.zeros(D)
    
    def momentum(self, momentum, grad, beta):
        return beta*momentum + (1-beta)*grad
    
    def update(self, beta, grad, lr):
        return beta - lr * grad
    
    def shuffle_data(self, x, y):
        idx = np.random.permutation(x.shape[0])
        return x[idx], y[idx]
    
    def fit_gradient_descent(self, x, y, nb_epochs, lr):
        losses = []
        beta = self.initialize(x)
        for epoch in range(nb_epochs+1):
            y_pred = self.linear_function(x, beta)
            loss = self.mean_square_error(y, y_pred)
            grad = self.gradient(x, y, beta)
            beta = self.update(beta, grad, lr)
            losses.append(loss)
            
            if epoch%5 == 0:
                print(f"Epoch {epoch}: loss {loss}")
            # self.plot(x, y, beta, epoch, plot_every=2)


        plt.plot(losses)
        plt.title("Loss for Linear Regression with GD") 
        plt.legend()
        plt.show()

    
    def fit_stochastic_gradient_descent(self, x, y, nb_epochs, lr):
       
        losses = []
        beta = self.initialize(x)
        
        for epoch in range(nb_epochs):
            runing_loss = 0
            momentum = 0.0
            x_shuffle, y_shuffle = self.shuffle_data(x, y)
            
            for idx in range(x_shuffle.shape[0]):
                sample_x = x_shuffle[idx].reshape(-1,x.shape[1])
                sample_y = y_shuffle[idx].reshape(-1,1)
                y_pred = self.linear_function(sample_x, beta)
                loss = self.mean_square_error(sample_y, y_pred)
                runing_loss += loss
                grads = self.gradient(sample_x, sample_y, beta)
                momentum = self.momentum(momentum, grads, beta)
                beta = self.update(beta, momentum, lr)
            
            average_loss = runing_loss / x.shape[0]
            losses.append(average_loss)
            if epoch%5 == 0:
                print(f"Epoch {epoch} loss {average_loss}")
            
        plt.plot(losses)
        plt.title("Loss for Linear Regression with SGD") 
        plt.legend()
        plt.show()
       
    
    def fit_mini_batch_gradient_descent(self, x, y, batch_size, nb_epochs, lr):
        losses = []
        nb_batches = x.shape[0]//batch_size
        beta = self.initialize(x)
        x, y = self.shuffle_data(x,y)
        
        for epoch in range(nb_epochs):
            runing_loss = 0
            
            for batch_idx in range(0, x.shape[0], batch_size):
                x_batch = x[batch_idx: batch_idx + batch_size]
                y_batch = y[batch_idx: batch_idx + batch_size]
                
                y_pred = self.linear_function(x_batch, beta)
                loss = self.mean_square_error(y_batch, y_pred)
                runing_loss += (loss*x_batch.shape[0])
                grads = self.gradient(x_batch, y_batch, beta)
                beta = self.update(beta, grads, lr)
                
            average_loss = runing_loss / x.shape[0]
            losses.append(average_loss)
            if epoch%5 == 0:
                print(f"Epoch {epoch} loss {average_loss}")
        
        plt.plot(losses)
        plt.title("Loss for Linear Regression with MB_SGD") 
        plt.legend()
        plt.show()
    