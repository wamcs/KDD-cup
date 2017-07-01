import pandas as pd
import numpy as np
import split as sp
from datetime import datetime,timedelta
import split as split
import csv
import xgboost as xgb
import operator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.svm import SVR
import os

splitDataPath = '../dataProcessing/splitData/splitData.csv'
splitTestDataPath = '../dataProcessing/splitData/testData.csv'
testDataPath = '../dataSets/testing_phase1/trajectories(table 5)_test1.csv'
testWeatherPath = '../dataSets/testing_phase1/weather (table 7)_test1.csv'
table3 = '../dataSets/training/links (table 3).csv'
table4 = '../dataSets/training/routes (table 4).csv'
table7 = '../dataProcessing/splitData/weather.csv'
# some data lack link
linkLen ={'A':{'2':10,'3':12},'B':{'1':13,'3':9},'C':{'1':16,'3':12}}
ignore = ['2016-10-1','2016-10-2','2016-10-3','2016-10-4','2016-10-5','2016-10-6','2016-10-7','2016-9-15''2016-9-16','2016-9-17']


def splitTestData():
    data = pd.read_csv(splitTestDataPath)
    data['starting_time'] = data['starting_time'].apply(lambda x: x.split(' ')[0])
    return data

def getTrainData():
    data = pd.read_csv(splitDataPath)
    data['starting_time'] = data['starting_time'].apply(lambda x: x.split(' ')[0])
    return data


def trainVelocityRegression(xTrain,yTrain,n = 400):
    xlf = xgb.XGBRegressor(learning_rate=0.1,
                        n_estimators=n,
                        silent=1,
                        objective='reg:linear',
                        nthread=-1,
                        subsample=1,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)
    xlf.fit(xTrain,yTrain)
    return xlf

# def trainVelocityRegression(xTrain,yTrain):
#     clf = ExtraTreesRegressor(n_estimators=400,min_samples_split=1,random_state=0,n_jobs = -1)
#     X_normalized = preprocessing.normalize(xTrain, norm='l2')
#     train = preprocessing.scale(X_normalized)
#     clf.fit(train,yTrain)
#     return clf

# def trainVelocityRegression(xTrain,yTrain):
#     clf = RandomForestRegressor(n_estimators=400,min_samples_split=1,random_state=0,n_jobs = -1)
#     X_normalized = preprocessing.normalize(xTrain, norm='l2')
#     train = preprocessing.scale(X_normalized)
#     clf.fit(xTrain,yTrain)
#     return clf


def handleVelData(data,isTest = False):

    handleData = data.copy()

    # ga
    if not isTest:
        time = pd.DataFrame(pd.date_range(start = '20160719000000',end = '20161017234000',freq = '20min'),columns=['time'])
    else:
        temp1 = pd.DataFrame(pd.date_range(start = '20161018060000',end = '20161018074000',freq = '20min'),columns=['time'])
        temp2 = pd.DataFrame(pd.date_range(start = '20161018150000',end = '20161018164000',freq = '20min'),columns=['time'])
        times = []
        times.append(temp1.copy())
        times.append(temp2.copy())
        for i in range(6):
            temp1['time'] = temp1['time'] + pd.DateOffset(days = 1)
            temp2['time'] = temp2['time'] + pd.DateOffset(days = 1)
            times.append(temp1.copy())
            times.append(temp2.copy())

        time = pd.concat(times)
        time.reset_index()

    time['week'] = time['time'].apply(lambda x: x.weekday())
    time['time_window'] = time['time'].apply(lambda x: x.hour*3+x.minute/20)
    time['time'] = time['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    result = pd.merge(time,handleData,how='left',on = ['time','week','time_window'])
    result = result.interpolate()
    result = result.fillna(method = 'bfill')
    result = result.fillna(method = 'pad')

    vel = list(result['travel_velocity'].values)
    vel.insert(0,-1)
    vel.remove(vel[-1])
    result['lp1'] = pd.Series(vel)
    vel.insert(0,-1)
    vel.remove(vel[-1])
    result['lp2'] = pd.Series(vel)
    vel.insert(0,-1)
    vel.remove(vel[-1])
    result['lp3'] = pd.Series(vel)
    vel.insert(0,-1)
    vel.remove(vel[-1])
    result['lp4'] = pd.Series(vel)
    vel.insert(0,-1)
    vel.remove(vel[-1])
    result['lp5'] = pd.Series(vel)
    vel.insert(0,-1)
    vel.remove(vel[-1])
    result['lp6'] = pd.Series(vel)

    return result

def time_window(s):
    time = datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
    minute = time.minute
    hour = time.hour
    return hour*3+minute/20

def addLastVel(x):
    x = list(x)
    x.insert(0,-1)
    x.remove(x[-1])
    return pd.DataFrame(x)

def splitDataByLane(data,links,isTest = False):
#     label = ['intersection_id','tollgate_id','lane_id','starting_time','time_window','week','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation','travel_lane','travel_velocity','travel_length']

    label = ['lane_id','time','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation','travel_velocity']
    dataSet = data.copy()
    dataSet['travel_velocity'] = dataSet['travel_velocity'].apply(lambda x: np.array(x.split(';')).astype(float))
    dataSet['travel_lane'] = dataSet['travel_lane'].apply(lambda x: np.array(x.split(';')).astype(float))
    dataSet['travel_length'] = dataSet['travel_length'].apply(lambda x: np.array(x.split(';')).astype(float))
    dataSet['travel_seq'] = dataSet['travel_seq'].apply(lambda x: x.split(';'))
    dataSet.drop(['vehicle_id','total_velocity','total_length'],axis = 1,inplace = True)

    weather = []
    weaTmp = dataSet[['starting_time','time_window','week','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation']]
    for name,group in weaTmp.groupby(['starting_time','time_window','week']):
        temp = []
        temp.extend(group.values[0].tolist())
        weather.append(temp)
    wLabel = ['time','time_window','week','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation']
    weather = pd.DataFrame(weather,columns = wLabel)

    result = pd.concat(list(dataSet['travel_seq'].apply(lambda x: pd.DataFrame(x,columns=['lane_id']))))
    result['time'] = result['lane_id'].apply(lambda x: x.split('#')[1])
    result['week'] = result['time'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").weekday())
    # time_window at some seq have problem
    result['time_window'] = result['time'].apply(time_window)
    result['time'] = result['time'].apply(lambda x:x.split(' ')[0])
    result['lane_id'] = result['lane_id'].apply(lambda x: x.split('#')[0])
    result['travel_velocity'] = pd.concat(list(dataSet['travel_velocity'].apply(lambda x: pd.DataFrame(x))))
    result['travel_lane'] = pd.concat(list(dataSet['travel_lane'].apply(lambda x: pd.DataFrame(x))))
    result['travel_length'] = pd.concat(list(dataSet['travel_length'].apply(lambda x: pd.DataFrame(x))))
    result['last_t_vel1'] = pd.concat(list(dataSet['travel_velocity'].apply(addLastVel)))
    # result = result.groupby(['time','lane_id']).mean().reset_index()
    result = pd.merge(result,weather,how = 'left',on = ['time','time_window','week'])



    velDataByLane = result[['time','week','time_window','lane_id','travel_velocity']]
    velDataByLane = velDataByLane.groupby(['time','week','time_window','lane_id']).mean().reset_index()

    print result.info()
    result.dropna(inplace = True)
    result = result.reset_index()
    result.drop(['index'],axis = 1,inplace = True)

    velByLink = {}
    for lane in links.keys():
        temp = velDataByLane[:][velDataByLane['lane_id'] == lane]
        velByLink[lane] = handleVelData(temp.drop(['lane_id'],axis = 1),isTest)
        # print '========'
        # print velByLink[lane].info()

    print '==========================='
    print 'begain to split'
    print result.info()
    return result,velByLink


def predictVelocity(start,end,routes,links,weatherPath,timeWindows,trainData,testData,lackDay = []):
    if not os.path.isfile('../dataProcessing/splitData/splitTrainDataByLane.csv'):
        splitTrainDataByLane,velTrainDict = splitDataByLane(trainData,links)

        print '=============='
        print 'generate TrainData'

        splitTrainDataByLane['in_top'] = splitTrainDataByLane['lane_id'].apply(lambda x: int(links[x][2]))

        label = ['time','week','time_window','lane_id','lp1','lp2','lp3','lp4','lp5','lp6']
        tmpVel = []
        for item in splitTrainDataByLane[['time','week','time_window','lane_id']].values:
            temp = list(item)
            lane = temp[-1]
            index = (velTrainDict[lane]['time'] == temp[0])&(velTrainDict[lane]['week'] == int(temp[1]))&(velTrainDict[lane]['time_window']==int(temp[2]))
            vel = list(velTrainDict[lane][['lp1','lp2','lp3','lp4','lp5','lp6']][index].values[0])
            temp.extend(vel)
            tmpVel.append(temp)

        tmpVel = pd.DataFrame(tmpVel,columns=label)
        # splitTrainDataByLane = pd.merge(splitTrainDataByLane,tmpVel,how='left',on = ['time','week','time_window','lane_id'])
        splitTrainDataByLane = pd.concat([splitTrainDataByLane,tmpVel[['lp1','lp2','lp3','lp4','lp5','lp6']]],axis=1)

        print '==========='
        print splitTrainDataByLane.info()
        print splitTrainDataByLane
        splitTrainDataByLane.to_csv('../dataProcessing/splitData/splitTrainDataByLane.csv',index = False)
    else:
        splitTrainDataByLane = pd.read_csv('../dataProcessing/splitData/splitTrainDataByLane.csv')

    # index = (~splitTrainDataByLane['time'].isin(ignore))&(splitTrainDataByLane['travel_velocity']<=30)&(splitTrainDataByLane['last_t_vel1']<=30)
    # splitTrainDataByLane = splitTrainDataByLane[:][index]

    eTrain = splitTrainDataByLane[:][splitTrainDataByLane['time'] < '2016-10-1']
    eTest = splitTrainDataByLane[:][splitTrainDataByLane['time'] >= '2016-10-1']

    y = eTrain['travel_velocity'].values
    eTrain.drop(['travel_velocity','lane_id','time','in_top'],axis = 1,inplace = True)
    x = eTrain.values

    eXlf = trainVelocityRegression(x,y,200)
    predictTest = eXlf.predict(eTest.drop(['travel_velocity','lane_id','time','in_top'],axis = 1).values)
    error = eTest['travel_velocity'].values - predictTest
    print error

    x = eTest.drop(['travel_velocity','lane_id','time','in_top'],axis = 1).values

    errorXlf = SVR().fit(x,error)
    # ==================================================================

    y = splitTrainDataByLane['travel_velocity'].values
    splitTrainDataByLane.drop(['travel_velocity','lane_id','time','in_top'],axis = 1,inplace = True)
    print splitTrainDataByLane.info()
    x = splitTrainDataByLane.values

    print '====='
    xlf = trainVelocityRegression(x,y)
    print '====='


    splitTestDataByLane,velTestDict = splitDataByLane(testData,links,True)
    print splitTestDataByLane.info()

    weather = []
    with open(weatherPath,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            weather.append(line[:])
    weather = np.array(weather)


    startTime = datetime.strptime(start,"%Y-%m-%d")
    endTime = datetime.strptime(end,"%Y-%m-%d") + timedelta(days = 1)
    times = []
    weeks = []
    tempTime = startTime
    while tempTime != endTime:
        times.append(tempTime.strftime("%Y-%m-%d"))
        weeks.append(tempTime.weekday())
        tempTime += timedelta(days = 1)

    print '====='
    data = []
    label = ['intersection_id','tollgate_id','time','lane_id','time_window','week','length','lane','last_t_vel1','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation','lp1','lp2','lp3','lp4','lp5','lp6','lw']
    for intersection in routes.keys():
        for tollgate in routes[intersection].keys():
            for i,time in enumerate(times):
                lvel = []
                for window in timeWindows:
                    if window == 24 or window == 51:
                        k = 0
                    if window == 51:
                        data.append(lvel)
                        lvel = []
                    rvel = []
                    for j,link in enumerate(routes[intersection][tollgate]):
                        if time in lackDay:
                            continue
                        weatherList = weather[np.nonzero((weather[:,0] == time)&(weather[:,1] == str((int(window)/9)*3)))[0],2:].astype(float).tolist()
                        if weatherList == []:
                            continue
                        temp = []
                        temp.append(intersection)
                        temp.append(tollgate)
                        temp.append(time)
                        temp.append(link)
                        temp.append(int(window))
                        temp.append(weeks[i])
                        temp.extend(links[link][0:-1])
                        if links[link][-1] == 0:
                            temp.append(-1)
                        else:
                            temp.append(0)
                        temp.extend(weatherList[0])
                        # temp.append(links[link][-1])
                        index = (velTestDict[link]['time'] == time)&(velTestDict[link]['week'] == weeks[i])&(velTestDict[link]['time_window']==int(window)-k-1)
                        vel = list(velTestDict[link][['travel_velocity','lp1','lp2','lp3','lp4','lp5']][index].values[0])
                        for s in range(k):
                            vel.insert(0,0)
                            vel.remove(vel[-1])
                        temp.extend(vel)
                        rvel.append(temp)
                    lvel.append(rvel)
                    k += 1
                data.append(lvel)

    print '=========='
    print data[0][0][0]

    for i in range(17):
        train = []
        trainNum = []
        for l,item in enumerate(data):
            intersection = item[0][0][0]
            tollgate = item[0][0][1]
            # print intersection,tollgate
            if linkLen[intersection][tollgate]<i:
                continue
            for j in range(i+1):
                k = i - j
                if(k>linkLen[intersection][tollgate]-5) or (j>5):
                    continue

                # print "(",j,k,")"
                # print item[j][k]
                temp  = item[j][k][4:]
                train.append(temp)
                trainNum.append((l,j,k,linkLen[intersection][tollgate]))
        train = np.array(train).astype(float)

        # X_normalized = preprocessing.normalize(train, norm='l2')
        # train = preprocessing.scale(X_normalized)

        result = xlf.predict(train)
        error = errorXlf.predict(train)
        result = error + result
        for m,index in enumerate(trainNum):
            l = index[0]
            j = index[1]
            k = index[2]
            size = index[3] - 5
            data[l][j][k].append(result[m])
            if k < size:
                data[l][j][k+1][8] = result[m]
            if j<=5:
                for n in range(5-j):
                    data[l][j+n+1][k][16+n] = result[m]

    print '=========='
    print 'finish predicting'


    predict = []
    for item in data:
        for row in item:
            for l in row:
                temp = l[0:3]
                temp.append(l[4])
                temp.append(l[-1])
                predict.append(temp)

    predict = pd.DataFrame(predict,columns = ['intersection_id','tollgate_id','starting_time','time_window','predict_velocity'])


    print '====='
    print predict.info()



    result = []
    for name,group in predict.groupby(['intersection_id','tollgate_id','starting_time','time_window'])['predict_velocity']:
        temp = list(name)
        velocitys = []
        for v in group:
            velocitys.append(v)
        velocitys = np.array(velocitys)
        temp.append(velocitys)
        result.append(temp)

    return pd.DataFrame(result,columns = ['intersection_id','tollgate_id','starting_time','time_window','predict_velocity'])

def predict(start,end,weatherPath,time,testData,trainData):
    routes = {}
    links = {}
    lengths = {}
    lengthsList = []
    with open(table3,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            # length,lane
            i = 0
            l = line[4].split(',')
            if len(l) == 1:
                if l[0] == '':
                    i = 0
                else:
                    i = 1
            else:
                i = 2
            links[line[0]] = [int(line[1]),int(line[3]),i]

    with open(table4,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            if line[0] not in routes.keys():
                routes[line[0]] = {}
                lengths[line[0]] = {}
            routes[line[0]][line[1]] = line[2].split(',')
            temp = []
            for i in routes[line[0]][line[1]]:
                temp.append(links[i][0])
            lengths[line[0]][line[1]] = np.array(temp)


    for i in lengths.keys():
        for j in lengths[i].keys():
            temp = []
            temp.append(i)
            temp.append(j)
            temp.append(lengths[i][j])
            lengthsList.append(temp)


    # print '==========================================='
    # print 'routes is',routes
    # print '==========================================='
    # print 'link is',links
    # print '==========================================='
    # print 'lengths is',lengths

    predictData = predictVelocity(start,end,routes,links,weatherPath,time,trainData,testData)
    print '==========================================='
    print 'predictData is',predictData

    lengths = pd.DataFrame(lengthsList,columns = ['intersection_id','tollgate_id','lengths'])
    predictData = pd.merge(predictData,lengths,how = 'left',on = ['intersection_id','tollgate_id'])

    preVel = predictData['predict_velocity'].values
    # hisVel = data['his_velocity'].values
    linkLength = predictData['lengths'].values
    print '==========================================='
    # print 'preVel',preVel
    # # print 'hisVel',hisVel
    # print 'linkLength',linkLength
    # vel = (preVel+hisVel)/2
    vel = preVel
    travelTime = []
    for i,v in enumerate(vel):
        travelTime.append(np.sum(linkLength[i]/v))

    travelTime = np.array(travelTime)

    predictData.insert(6,'avg_travel_time',travelTime)
    predictData.drop(['predict_velocity','lengths'],axis = 1,inplace = True)
    print '==========================================='
    print 'data is ',predictData
    return predictData

def task1():
    TrainData = getTrainData()
    TestData = splitTestData()

    print 'handle data'
    timeWindows = range(24,30)
    timeWindows.extend(range(51,57))

    print 'begain predicting'
    predictData = predict('2016-10-18','2016-10-24',testWeatherPath,timeWindows,TestData,TrainData)
    formatTrans(predictData)

def formatTrans(predictData):

    predictData['time_window'] = predictData['time_window'].apply(lambda x: int(x))
    # predictData['avg_travel_time'] = predictData['avg_travel_time'].apply(lambda x: round(x,2))
    firstTime = range(24,30)
    secondTime = range(51,57)
    first = predictData[:][predictData['time_window'].isin(firstTime)]
    second = predictData[:][predictData['time_window'].isin(secondTime)]
    predictData = pd.concat([first,second])

    time = []
    days = predictData['starting_time'].values
    timeWindows = predictData['time_window'].values

    predictData.drop(['starting_time','time_window'],axis = 1,inplace = True)
    for i,day in enumerate(days):
        day = datetime.strptime(day,"%Y-%m-%d")
        minute = int(timeWindows[i])*20
        hour = minute/60
        minute = minute - hour*60
        day = day.replace(hour = hour,minute = minute)
        start = day
        end = day + timedelta(minutes = 20)
        time.append("["+start.strftime("%Y-%m-%d %H:%M:%S")+","+end.strftime("%Y-%m-%d %H:%M:%S")+")")
    time = np.array(time)
    predictData.insert(2,'time_window',time)
    predictData.to_csv('../dataProcessing/answerSVR.csv',index = False)



if __name__ == '__main__':
    task1()
