# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:26:00 2022

@author: znhao
"""
from scipy.optimize import linprog
import numpy as np

import utils
import MyRegressor

data = utils.prepare_data_gaussian()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

sol = MyRegressor.MyRegressor(alpha=0.00)
train_error = sol.train(trainX, trainY)
pred_testY,test_error = sol.evaluate(testX, testY)

n_zeros = np.count_nonzero(sol.weight==0)

# Y = trainY
# X = trainX
# N = trainX.shape[0]
# M = trainX.shape[1]

## variables: t(N,1), u(M,1), theta(M,1), b(1,1)
## parameters: X(N,M), Y(N,1)
# alpha = 0

# c = np.hstack((1/N*np.ones([1,N]),alpha*np.ones([1,M]), np.zeros([1,M+1])))
# A = np.block([
#               [-np.eye(N), np.zeros([N,M]), -X, -np.ones([N,1])], 
#               [-np.eye(N), np.zeros([N,M]), X, np.ones([N,1])],
#               [np.zeros([M,N]), -np.eye(M), np.eye(M), np.zeros([M,1])],
#               [np.zeros([M,N]), -np.eye(M), np.eye(M), np.zeros([M,1])]
#               ])

# b = np.hstack((-Y, Y, np.zeros([M]), np.zeros([M])))

# sol = linprog(c, A_ub=A, b_ub=b)
# weight = sol.x[N+M:-1]
# bias = sol.x[-1]


# c = np.hstack((1/N*np.ones([1,N]), np.zeros([1,M+1])))
# A = np.block([
#              [-np.eye(N), -X, -np.ones([N,1])], 
#              [-np.eye(N), X, np.ones([N,1])]
#              ])

# b = np.hstack((-Y, Y))

# sol = linprog(c, A_ub=A, b_ub=b)
# weight = sol.x[N:-1]
# bias = sol.x[-1]