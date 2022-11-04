import numpy as np
import utils
from MyRegressor import MyRegressor

data = utils.prepare_data_gaussian()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

percentage_list = [0.01,0.1,0.3,0.5,1]
train_err  = np.zeros([len(percentage_list),1])
test_err  = np.zeros([len(percentage_list),1])
feat_num  = np.zeros([len(percentage_list),1])

c = 0
    
for percentage in percentage_list:
    
    sol = MyRegressor(alpha=0)
    selected_feat = sol.select_features(trainX, trainY, percentage)
    feat_num[c] = int(trainX.shape[1]*percentage)
    train_err[c] = sol.train(trainX[:,selected_feat], trainY)
    pred_testY,test_err[c] = sol.evaluate(testX[:,selected_feat], testY)
    c += 1

result = {'taskID':'1-3', 'feat_num':percentage_list, 'train_err':train_err, 'test_err':test_err}
utils.plot_result(result)
