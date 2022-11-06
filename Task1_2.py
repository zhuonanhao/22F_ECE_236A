import numpy as np
import utils
from MyRegressor import MyRegressor

# data = utils.prepare_data_gaussian()
data = utils.prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

alpha_list = np.linspace(0,1,101)
train_err  = np.zeros([len(alpha_list),1])
test_err  = np.zeros([len(alpha_list),1])
num_zeros  = np.zeros([len(alpha_list),1])

c = 0
    
for alpha in alpha_list:
    
    sol = MyRegressor(alpha=alpha)
    train_err[c] = sol.train(trainX, trainY)
    pred_testY,test_err[c] = sol.evaluate(testX, testY)
    num_zeros[c] = np.count_nonzero(sol.weight==0)
    c += 1

result = {'taskID':'1-2', 'alpha':alpha_list, 'train_err':train_err, 'test_err':test_err}
utils.plot_result(result)