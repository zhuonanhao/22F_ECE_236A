import numpy as np
import utils
from MyRegressor import MyRegressor

data = utils.prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']


# we simulate the online setting by handling training data samples one by one
cost = 0.9
maxCost = 0.9
sent_databyte = 0
p_feature = 0.5

selected_feat = [i for i in range(58)]

for index, x in enumerate(trainX):
    
    # Sensor collects data sample
    y = trainY[index]
    
    # Append current data sample to history dataset
    if index == 0:
        histX = np.array([x])
        histY = y
    else:
        histX = np.vstack([histX, x])
        histY = np.vstack([histY, y])
        
    # Collected databyte
    collected_databyte = histX.size + histY.size
    
    # Decide whether send data to central
    current_cost = sent_databyte/collected_databyte
    
    if current_cost <= maxCost:
        
        # Send selected data to centrl node
        selected_trainX = histX[:,selected_feat]        
        sent_databyte = selected_trainX.size
        
        # Regressor initilization
        sol = MyRegressor(alpha=0)
        
        # Feature selection
        selected_feat = sol.select_features(trainX, trainY, p_feature)
        
    else:
        
        p_feature = cost
    

    

    
    
    
    
    
