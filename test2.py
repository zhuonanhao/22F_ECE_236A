import numpy as np
import utils
from MyRegressor import MyRegressor
from sklearn.feature_selection import SelectKBest, f_regression


data = utils.prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']


# we simulate the online setting by handling training data samples one by one
cost = 0.75
maxCost = cost
p_feature = 0.5
err_threshold = 0.5
c = 0
train_err  = np.zeros([trainX.shape[0],1])
test_err  = np.zeros([trainX.shape[0],1])

for index, x in enumerate(trainX):
    
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
            train_err = sol.train(sent_trainX, sent_trainY)
            weight = sol.weight
            bias = sol.bias
            predY, test_err = sol.evaluate(testX, testY)
            print(x_err,index, train_err, test_err)
        
    

    

    
    
    
    
    
