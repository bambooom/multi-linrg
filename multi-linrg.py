# -*- coding: utf-8 -*-
#!/usr/bin/env python
# author: @bambooom

'''
multivariate linear regression:
    y = w0 + w1 * x1 + w2 * x2 + w3 * x3

Use two methods to calculate w0,w1,w2,w3:
    1. Explicit analytical solution by matrix: ols_matrix
    2. stochastic gradient descent: sgd
'''

import numpy as np

def ols_matrix(x,y):
    '''
    Estimate w by minimizing sum of squared residuals
        x: indep variable matrix with column x0,x1,x2... (x0=1 for intercept)
        y: dependent variable column
    w_array = (x.transpose*x)^(-1)*x.transpose*y
    '''

    from numpy.linalg import inv
    return inv(np.dot(x.T,x)).dot(x.T).dot(y)

def cost_func(ws,x,y): #cost function J(w)
    return sum((np.dot(x,ws)-y)**2)/2

def sgd(ws,alpha,x,y,iter_num=100):
    '''
    start initial guess ws, update w by small step
        ws: initial guess of w
        alpha: learning rate
        x: indep variable matrix with column x0,x1,x2... (x0=1 for intercept)
        y: dependent variable column
        iter_num: number of iteration,default =100
    '''

    for j in xrange(iter_num):
        for i in xrange(len(y)): # iteration over training data
            old_cost=cost_func(ws,x,y)
            pred_y = np.dot(x,ws) # predicted y value
            error = y - pred_y # error in y
            ws = ws + alpha*error[i]*x[i,:] # update w
            new_cost=cost_func(ws,x,y)

            if (new_cost > old_cost): # if cost not decreased, back to last step
                ws = ws - alpha*error[i]*x[i,:]
    return ws

if __name__ == '__main__':
    # read data from csv
    raw = np.genfromtxt ('data.csv', delimiter=",",skip_header=1)
    y = raw[:,-1] # last column is y value
    x_matrix = np.append(np.ones(len(y)).reshape(len(y),1),raw[:,0:-1],axis=1)
    row,col=x_matrix.shape

    ws = np.ones(col) # initial guess by [1,1,1,1]
    alpha = 0.001 #learning rate

    w_1 = ols_matrix(x_matrix,y)
    w_2 = sgd(ws,alpha,x_matrix,y)

    print "1st method - analytical solution by matrix"
    for i, w_1[i] in enumerate(w_1):
        print "w{} = {:f}".format(i,w_1[i])

    print "2nd method - stochastic gradient descent"
    for i, w_2[i] in enumerate(w_2):
        print "w{} = {:f}".format(i,w_2[i])
