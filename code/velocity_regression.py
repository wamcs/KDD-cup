# -*- coding utf-8 -*-

import numpy as np
import xgboost as xgb
import csv
from datetime import datetime

path = '../dataProcessing/splitData/splitData.csv'
pathVelocity = '../dataProcessing/linkVolecity/LinkVolecity.csv'

def readData():
    # label = ['time','week','wind_direction','wind_speed','temperature','rel_humidity','precipitation','length','lane','objective_velocity']
    # with open(pathVelocity,'r') as fr:
    #     lines = csv.reader(fr)
    #     data = {}
    #     for line in lines:
    #         if lines.line_num == 1:
    #             continue
    #         data[line[0]] = float(line[1])

    yTrainData = []
    trainData = []
    yTestData = []
    testData = []
    allData = []
    with open(path,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            leap = False

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
            for v in velocitys:
                if v>30:
                    leap = True
            if leap:
                continue

            for i in range(len(lanes)):
                temp = []
                temp.append(time_window)
                temp.append(week)
                temp.extend(weather)
                temp.append(lengths[i])
                temp.append(lanes[i])
                if i>0:
                    temp.append(velocitys[i-1])
                else:
                    temp.append(-1)
                if time.month == 10 and time.day>=11:
                    testData.append(temp)
                    yTestData.append(velocitys[i])
                else:
                    trainData.append(temp)
                    yTrainData.append(velocitys[i])
                allData.append(temp)
    trainData = np.array(trainData)
    yTrainData = np.array(yTrainData)
    testData = np.array(testData)
    yTestData = np.array(yTestData)
    allData = np.array(allData)
    maxVec = np.max(allData,axis=0).astype(float)
    trainData = trainData/maxVec
    testData = testData/maxVec
    print trainData.shape
    print testData.shape
    return trainData,yTrainData,testData,yTestData

def main():
    x,y,x_test,y_test = readData()

    xlf = xgb.XGBRegressor(learning_rate=0.1,
                        n_estimators=600,
                        silent=1,
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

    print 'error is %f' %(np.sum(np.abs(test-y_test)/y_test)/y_test.shape[0])
    print 'min error is %f'%(min(test-y_test))

    minValue = min(test)
    maxValue = max(test)
    a = y_test[np.nonzero((y_test <= maxValue)&(y_test>=minValue))]
    print float(a.shape[0])/float(y_test.shape[0])

if __name__ == '__main__':
    main()
