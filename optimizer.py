import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tools import inverse

class LSE():
    def fit(self, X, y, lambd=0):
        xtx = np.dot(X.T, X)
        xtx = xtx + lambd * np.identity(len(xtx))
        xtx = inverse(xtx)
        xtxxt = np.dot(xtx, X.T)
        self.theta = np.dot(xtxxt, y)
        
        y_pred = self.predict(X)
        self.loss = self.loss_function(y, y_pred)
    
    def loss_function(self, y, y_pred):
        error = sum((y - y_pred)**2)
        return error
    
    def predict(self, X):
        return np.dot(X, self.theta)
    
    def fitting_line(self, n):
        X_term = ['X^%d'%i for i in range(n-1, 0, -1)] + ['']
        term = ' + '.join(['%f%s'%(self.theta[i], X_term[i]) for i in range(n)])
        return term
    
    def show_report(self, X, y):
        n = X.shape[1]
        print('\nLSE:')
        print('Fitting line: %s'%self.fitting_line(n))
        print('Total error: %f'%self.loss)


class Newton():
    def __init__(self, maxiter=100):
        self.maxiter = maxiter
        
    def fit(self, X, y):
        self.n = X.shape[1]
        self.theta = np.zeros((X.shape[1], 1))
        y_pred = self.predict(X)
        
        loss = self.loss_function(y, y_pred)
        m, n = X.shape

        for i in range(0, self.maxiter):
            gradient = 2 * np.dot(np.dot(X.T, X), self.theta) - (2 * np.dot(X.T, y)).reshape(n, 1)
            hessian = 2 * np.dot(X.T, X)
            self.theta = self.theta - np.dot(inverse(hessian), gradient)

            y_pred = self.predict(X)
            new_loss = self.loss_function(y, y_pred)
            error = new_loss - loss
            if abs(error) < 0.05:
                break
            else:
                self.loss = new_loss
    
    def loss_function(self, y, y_pred):
        loss = np.sum((y - y_pred)**2)
        return loss
    
    def predict(self, X):
        return np.dot(X, self.theta).reshape(-1)
    
    def fitting_line(self):
        X_term = ['X^%d'%i for i in range(self.n-1, 0, -1)] + ['']
        term = ' + '.join(['%f%s'%(self.theta[i], X_term[i]) for i in range(self.n)])
        return term
    
    def show_report(self):
        print('\nNewton\'s Method:')
        print('Fitting line: %s'%self.fitting_line())
        print('Total error: %f'%self.loss)