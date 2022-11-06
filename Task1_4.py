import numpy as np
import utils
from MyRegressor import MyRegressor

# data = utils.prepare_data_gaussian()
data = utils.prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

percentage_list = np.linspace(0.01,1,100)
train_err  = np.zeros([len(percentage_list),1])
test_err  = np.zeros([len(percentage_list),1])

c = 0
    
for percentage in percentage_list:
    
    sol = MyRegressor(alpha=0)
    selected_trainX, selected_trainY = sol.select_sample(trainX, trainY, percentage)
    train_err[c] = sol.train(selected_trainX, selected_trainY)
    pred_testY,test_err[c] = sol.evaluate(testX, testY)
    c += 1

result = {'taskID':'1-4', 'sample_num':percentage_list, 'train_err':train_err, 'test_err':test_err}

utils.plot_result(result)