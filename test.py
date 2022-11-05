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
cost = 0.9
maxCost = cost
sent_volume = 0
p_feature = 0.5

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
        
        # Initilize fitting parameters
        weight = np.zeros(x.shape[0])
        bias = 0
        
    else:
        
        # Updata history database
        histX = np.vstack([histX, x])
        histY = np.append(histY, y)
        
    # Collected data volume (maximum set that can be sent)
    collected_volume = histX.size + histY.size
    
    if index >= 3:
        
        # Select best features from history database
        p_feat = maxCost
        
        feat_num = int(histX.shape[1]*p_feat)
        selector = SelectKBest(f_regression, k=feat_num)
        selector.fit_transform(histX, histY)
        feat_bool = selector.get_support()
        selected_feat = [i for i, x in enumerate(feat_bool) if x]
        
        # Prepare sent data
        sent_trainX = histX[:,selected_feat]
        sent_trainY = histY
        
        # Sent data volume
        sent_volume = sent_trainX.size + sent_trainY.size
        
        if sent_volume <= collected_volume * maxCost:
            
            # Central node do:
            sol = MyRegressor(alpha=0)
            sol.train(sent_trainX, sent_trainY)
            weight = sol.weight
            bias = sol.bias
    

    

    
    
    
    
    
