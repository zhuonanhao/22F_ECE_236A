import numpy as np
import utils
from MyRegressor import MyRegressor

data = utils.prepare_data_gaussian()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

p_feature_list = np.linspace(0.1,1,10)
p_sample_list = np.linspace(0.1,1,10)
train_err  = np.zeros([len(p_feature_list)*len(p_sample_list),1])
test_err  = np.zeros([len(p_feature_list)*len(p_sample_list),1])
cost  = np.zeros([len(p_feature_list)*len(p_sample_list)])

c = 0
    
for p_feature in p_feature_list:
    for p_sample in p_sample_list:
        
        cost[c] = p_sample*p_feature
        sol = MyRegressor(alpha=0)
        selected_trainX, selected_trainY, selected_feat = sol.select_data(trainX, trainY, p_feature, p_sample)
        train_err[c] = sol.train(selected_trainX, selected_trainY)
        pred_testY,test_err[c] = sol.evaluate(testX[:,selected_feat], testY)
        c += 1

seq = np.argsort(cost)
result = {'taskID':'1-5', 'cost':cost[seq], 'train_err':train_err[seq], 'test_err':test_err[seq]}

utils.plot_result(result)

# p_feature = 0.9
# p_sample = 0.9

# sol = MyRegressor(alpha=0)

# selected_feat = sol.select_features(trainX, trainY, p_feature)
# temp_trainX, temp_trainY = sol.select_sample(trainX, trainY, p_sample)


# selected_trainX = temp_trainX[:,selected_feat]
# selected_trainY = temp_trainY