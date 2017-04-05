# -*- coding utf-8 -*-

import numpy as np
import xgboost as xgb
import
from datetime import datetime
import matplotlib.pyplot as plt

path = '../dataProcessing/splitData/splitData.csv'
pathVelocity = '../dataProcessing/linkVolecity/LinkVolecity.csv'

def readData():
    # label = ['time','week','wind_direction','wind_speed','temperature','rel_humidity','precipitation','length','lane','objective_velocity']
    with open(pathVelocity,'r') as fr:
        lines = csv.reader(fr)
        data = {}
        for line in lines:
            if lines.line_num == 1:
                continue
            data[line[0]] = float(line[1])

    yTrainData = []
    trainData = []
    yTestData = []
    testData = []
    with open(path,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            if line[8] == '':
                continue
            weather = [float(i) for i in line[10:15]]

            time = datetime.strptime(line[3], "%Y-%m-%d %H:%M:%S")
            # use 0-71 represent time_window
            time_window = (time.hour * 60 + time.minute)/20
            # 0 - 6 : Mon - Sun
            week = time.weekday()

            lanes =[int(i) for i in line[15].split(';')]
            velocitys = [float(i) for i in line[16].split(';')]
            lengths = [int(i) for i in line[17].split(';')]
            linkId = [item.split('#')[0] for item in line[4].split(';')]
            for i in range(len(lanes)):
                temp = []
                temp.append(time_window)
                temp.append(week)
                temp.extend(weather)
                temp.append(lengths[i])
                temp.append(lanes[i])
                temp.append(data[linkId[i]])
                if time.month == 10 and time.day>=11:
                    testData.append(temp)
                    yTestData.append(velocitys[i])
                else:
                    trainData.append(temp)
                    yTrainData.append(velocitys[i])
    return trainData,yTrainData,testData,yTestData

def main():
    trainData,yTrainData,testData,yTestData = readData()
    x = np.array(trainData)
    y = np.array(yTrainData)

    x_test = np.array(testData)
    y_test = np.array(yTestData)

    xlf = xgb.XGBRegressor(learning_rate=0.1,
                        n_estimators=200,
                        silent=0,
                        objective='reg:linear',
                        nthread=-1,
                        subsample=1,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)

    xlf.fit(x,y)
    test = xlf.predict(x_test)

    # plt.plot(range(y_test.shape[0]),y_test,'bx',markersize = 10)
    # plt.plot(range(test.shape[0]),test,'r+',markersize = 10)
    # plt.show()

    # print 'error is %f' %(np.sum(np.abs(test-y_test)/y_test)/y_test.shape[0])
    # print 'min error is %f'%(min(test-y_test))

    minValue = min(test)
    maxValue = max(test)
    a = y_test[np.nonzero((y_test <= maxValue)&(y_test>=minValue))]
    print float(a.shape[0])/float(y_test.shape[0])

if __name__ == '__main__':
    main()
