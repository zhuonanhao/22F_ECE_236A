import numpy as np
import utils
from MyRegressor import MyRegressor

data = utils.prepare_data_gaussian()
# data = utils.prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

cost_list = np.linspace(0.1,1,10)
train_err = np.zeros([len(cost_list),1])
test_err = np.zeros([len(cost_list),1])
training_cost = np.zeros([len(cost_list),1])

c = 0
for cost in cost_list:
        
    sol = MyRegressor(alpha=0)
    training_cost[c], train_err[c] = sol.train_online(trainX, trainY, cost)
    pred_testY,test_err[c] = sol.evaluate(testX, testY)
    c += 1


result = {'taskID':'2', 'cost':cost_list, 'train_err':train_err, 'test_err':test_err}

utils.plot_result(result)