from sklearn.metrics import mean_absolute_error
import numpy as np
import random
from scipy.optimize import linprog

class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0   # N * N
        self.alpha = alpha
        
    def select_features(self, trainX, trainY, percentage):
        ''' Task 1-3
            Todo: '''
            
        feat_num = int(trainX.shape[1]*percentage)
        
        maxIter = 100
        minErr = 1000
        iter = 0
        while iter < maxIter:
            
            feat_ind = random.sample(range(500),feat_num)
            err = self.train(trainX[:,feat_ind], trainY)
            
            if err < minErr:
                selected_feat = feat_ind
                selected_feat.sort()
                
            iter += 1
        
        return selected_feat # The index List of selected features
        
        
    def select_sample(self, trainX, trainY):
        ''' Task 1-4
            Todo: '''
         
        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY):
        ''' Task 1-5
            Todo: '''
        
        return selected_trainX, selected_trainY
    
    
    def train(self, trainX, trainY):
        ''' Task 1-2
            Todo: '''
            
        
        Y = trainY
        X = trainX
        N = trainX.shape[0]
        M = trainX.shape[1]

        ## variables: t(N,1), theta(M,1), b(1,1)
        ## parameters: X(N,M), Y(N,1)
        
        c = np.hstack((1/N*np.ones([1,N]),self.alpha*np.ones([1,M]), np.zeros([1,M+1])))
        A = np.block([
                      [-np.eye(N), np.zeros([N,M]), -X, -np.ones([N,1])], 
                      [-np.eye(N), np.zeros([N,M]), X, np.ones([N,1])],
                      [np.zeros([M,N]), -np.eye(M), np.eye(M), np.zeros([M,1])],
                      [np.zeros([M,N]), -np.eye(M), np.eye(M), np.zeros([M,1])]
                      ])
        
        b = np.hstack((-Y, Y, np.zeros([M]), np.zeros([M])))
        
        sol = linprog(c, A_ub=A, b_ub=b)
        self.weight = sol.x[N+M:-1]
        self.bias = sol.x[-1]
        
        predY, train_error = self.evaluate(trainX, trainY)
        return train_error
    
    
    def train_online(self, trainX, trainY):
        ''' Task 2 '''

        # we simulate the online setting by handling training data samples one by one
        for index, x in enumerate(trainX):
            y = trainY[index]

            ### Todo:
            
        return self.training_cost, train_error

    
    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias