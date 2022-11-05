import numpy as np
import utils
from MyRegressor import MyRegressor

data = utils.prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

sol = MyRegressor(alpha=0)
train_err = sol.train_online(trainX, trainY, 0.2)
pred_testY,test_err = sol.evaluate(testX, testY)