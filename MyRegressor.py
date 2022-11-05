from sklearn.metrics import mean_absolute_error

import numpy as np
from scipy.optimize import linprog
from sklearn.feature_selection import SelectKBest, f_regression

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
        
        selector = SelectKBest(f_regression, k=feat_num)
        selector.fit_transform(trainX, trainY)
        feat_bool = selector.get_support()
        selected_feat = [i for i, x in enumerate(feat_bool) if x]
        
        return selected_feat # The index List of selected features
        
        
    def select_sample(self, trainX, trainY, percentage):
        ''' Task 1-4
            Todo: '''
            
        N = trainX.shape[0]
        sample_num = int(trainX.shape[0]*percentage)
        
        c = np.linalg.norm(trainX, ord=1, axis=1)
        A = np.block([
                      [np.ones([1,N])], 
                      [-np.ones([1,N])], 
                      [np.eye(N)],
                      [-np.eye(N)]
                      ])
        b = np.hstack((sample_num, -sample_num, np.ones([N]), np.zeros([N])))
        integrality = np.ones([N])
        sol = linprog(c, A_ub=A, b_ub=b, integrality=integrality)
        
        sample_bool = sol.x
        
        selected_sample = [i for i, x in enumerate(sample_bool) if x]
        selected_trainX = trainX[selected_sample,:]
        selected_trainY = trainY[selected_sample]
         
        return selected_trainX, selected_trainY    # A subset of trainX and trainY


    def select_data(self, trainX, trainY, p_feature, p_sample):
        ''' Task 1-5
            Todo: '''
        
        selected_feat = self.select_features(trainX, trainY, p_feature)
        temp_trainX, temp_trainY = self.select_sample(trainX, trainY, p_sample)
        
        
        selected_trainX = temp_trainX[:,selected_feat]
        selected_trainY = temp_trainY
        
        return selected_trainX, selected_trainY, selected_feat
    
    
    def train(self, trainX, trainY):
        ''' Task 1-2
            Todo: '''
        
        Y = trainY
        X = trainX
        N = trainX.shape[0]
        M = trainX.shape[1]
        
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
    
    
    def train_online(self, trainX, trainY, cost):
        ''' Task 2 '''
        
        maxCost = cost
        err_threshold = 0.5

        # we simulate the online setting by handling training data samples one by one
        for index, x in enumerate(trainX):
            y = trainY[index]

            ### Todo:   
            
            ### Sensor node do:
            
            # Sensor collects data sample
            y = trainY[index]
            
            # Append current data sample to history database
            if index == 0:
                
                # Initilize history database
                histX = np.array([x])
                histY = y
                
                sample_num = 0
                
                # Initilize fitting parameters
                weight = np.zeros(x.shape[0])
                bias = 0
                
            else:
                
                # Updata history database
                histX = np.vstack([histX, x])
                histY = np.append(histY, y)
                
            # Collected data volume (maximum set that can be sent)
            collected_volume = histX.size + histY.size
            
            x_err = abs(y-x.dot(weight)-bias)
            
            if index >= 2 and x_err > err_threshold:
                
                sample_num = np.append(sample_num,index)
                
                # Prepare sent data
                sent_trainX = histX[sample_num,:]
                sent_trainY = histY[sample_num]
                
                # Sent data volume
                sent_volume = sent_trainX.size + sent_trainY.size
                
                if sent_volume <= collected_volume * maxCost:
                    
                    # Central node do:
                    sol = MyRegressor(alpha=0)
                    train_error = sol.train(sent_trainX, sent_trainY)
                    weight = sol.weight
                    bias = sol.bias
            
        return self.training_cost, train_error

    
    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias